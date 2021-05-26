'''
Abandon hope all ye who enter here.

To do:

Code:
* create clean template notebook that walks through all the steps (both to develop above and have something to follow in the future)
* integrate remaining notebook functions: closed loop, cmat processing, IF extraction....
* figure out how to handle storing / referencing / accessing calibration products (interaction M, control M)
* figure out how to handle default settings (both instrument + optimization)
* command line functions for:
    * running FDPR on a given camera and DM w/ a predefined control matrix
    * measuring an interaction matrix
    * building a control matrix from an interaction matrix (how will you define the control modes?)

'''
from copy import deepcopy
from collections import OrderedDict
from functools import partial
import multiprocessing as mp
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

import numpy as np
try:
    import cupy as cp
except ImportError:
    logger.warning('Could not import cupy. You may lose functionality.')
    cp = None
from scipy.optimize import minimize
from skimage.transform import downscale_local_mean
from poppy.zernike import arbitrary_basis

from ..imutils import gauss_convolve, get_gauss, fft2_shiftnorm, ifft2_shiftnorm
from .. import pupils

#--------------HELPER FUNCTIONS----------------

def get_array_module(arr):
    if cp is not None:
        return cp.get_array_module(arr)
    else:
        return np

def cexp(arr):
    # faster than np.exp(1j*arr)
    xp = get_array_module(arr)
    return xp.cos(arr) + 1j*xp.sin(arr)

def param_dict_to_list(param_dict):
    '''
    This is intended to give you a mapping of parameters to fit
    and their parameter type.
    '''
    to_fit = []
    keys_to_fit = []
    for key, val in param_dict.items():
        if val[-1] is True:
            curval = list(val[0])
            to_fit.extend(deepcopy(curval))
            keys_to_fit.extend([key,]*len(curval))
    return to_fit, keys_to_fit
    
def update_param_dict_from_list(paramlist, x0, old_param_dict):
    updated_dict = deepcopy(old_param_dict)
    
    ukeys = np.unique(paramlist)
    for key in ukeys:
        updated_dict[key] = [np.asarray(x0, dtype=np.float32)[np.array(paramlist) == key], True]
    return updated_dict

def lateral_shift_to_defocus_rms(delz, fnum):
    return -delz / (28 * (fnum)**2 )

def defocus_rms_to_lateral_shift(drms, fnum):
    return  -drms * 28 * (fnum)**2

def get_lens_phase(pupil, f, wavelen, r_coords):
    return -2*np.pi/wavelen*(r_coords**2)/(2*f) * pupil

def my_fourier_shift(dely, delx, shape, xp=np, coords=None):
    if coords is None:
        cy, cx, r = get_coords(shape[0], 1, xp=xp) # assuming it's square
    else:
        cy, cx, r = coords
    return cexp(-2*np.pi*(dely*cy/shape[0] + delx*cx/shape[0]))#np.exp(-1j*2*np.pi*(dely*cy/shape[0] + delx*cx/shape[0]))

def get_coords(N, scale, xp=np, center=True):
    if center:
        cen = N/2.0
    else:
        cen = 0
    cy, cx = (xp.indices((N,N)) - cen) * scale
    r = xp.sqrt(cy**2 + cx**2)
    return [cy, cx, r]

def get_coords_fftshift(N, xp=np):
    freqs = xp.fft.fftfreq(N, d=1./N)
    fx, fy = xp.meshgrid(freqs, freqs)
    fr = xp.sqrt(fx**2 + fy**2)
    return [fy, fx, fr]

def center_of_mass(image):
    xp = get_array_module(image)
    
    indices = xp.indices(image.shape)
    return xp.average(indices[0], axis=None, weights=image), xp.average(indices[1], axis=None, weights=image)

def get_median(image, axis=None):
    if isinstance(image, cp.core.ndarray):
        image = cp.asnumpy(image)
    return np.median(image, axis=axis)

#--------------FORWARD MODEL----------------

def get_free_space_tf(z, wavelen, rcoords):
    xp = get_array_module(rcoords)
    pi = np.pi
    tf = cexp(2*np.pi*z[:,None,None]*xp.sqrt(1/wavelen**2 - rcoords**2))
    return tf
    
def propagate_by_angular_spectrum(Efield, wavelen, z, rcoords):
    Uf = fft2_shiftnorm(Efield, axes=(-2,-1), shift=False) # angular spectrum of input field    
    Uk = Uf * get_free_space_tf(z, wavelen, rcoords) # propagate to K plane
    Ek = ifft2_shiftnorm(Uk, axes=(-2,-1), shift=False) # back to K E field
    return Ek, Uk

def fresnel_propagate(Efield, wavelen, rcoord1, rcoord2, z0, to_focal_plane=False):
    # E field plane coords
    k = 2*np.pi/wavelen
    pi = np.pi
    
    if to_focal_plane:
        fresnel_phasor = 1 # reduces to fourier transform of Efield
    else:
        fresnel_phasor =  cexp(k/(2*z0)*rcoord1**2)

    return fft2_shiftnorm(Efield  * fresnel_phasor, shift=False)

def fraunhofer_simulate_psf(pupil, scale_factor, bg, phase, add_noise=True):
    pupil_field = pupil * cexp(phase)#np.exp(1j*phase)
    coh_psf = fft2_shiftnorm(pupil_field, shift=False)
    psf = (coh_psf * coh_psf.conj()).real
    #norm = np.sum(psf)
    
    psf_scaled = psf * scale_factor + bg
    if add_noise:
        psf_out = scale_and_noisify(psf_scaled, 1., 0.)
    else:
        psf_out = psf_scaled
    return psf_out

def simulate_psf(pupil, phase, wavelen, focal_length, coords1, coords2, to_focal_plane=False):
    xp = get_array_module(pupil)
    Epupil = pupil * cexp(phase)#np.exp(1j*phase) #ne.evaluate('pupil * exp(1j*phase)')
    norm = xp.sqrt(xp.sum(Epupil * Epupil.conj()).real)
    Epupil /= norm
    Efocal = fresnel_propagate(Epupil, wavelen, coords1[-1], coords2[-1], focal_length, to_focal_plane=to_focal_plane)
    return xp.abs(Efocal)**2, Efocal, Epupil #incoherent and coherent psf

def simulate_from_coeffs(pupil, zbasis, wavelen, focal_length, coords1, coords2, zcoeffs=None, pcoeffs=None, bcoeffs=None, static_phase=0, to_focal_plane=False):
    xp = get_array_module(pupil)
    phase = xp.zeros(pupil.shape)
    amplitude = xp.zeros(pupil.shape)
    if zcoeffs is not None: # zernike coeffs
        phase += xp.sum(zcoeffs[:,None,None]*zbasis[:len(zcoeffs)],axis=0) #xp.einsum('i,ijk->jk', zcoeffs, zbasis) #np.sum(zcoeffs[:,None,None]*zbasis[:len(zcoeffs)],axis=0)
    if pcoeffs is not None: # pixel by pixel values
        phase[pupil] += pcoeffs
    if bcoeffs is not None: # pixel by pixel amplitude values
        mn = xp.count_nonzero(pupil)
        acoeffs = amplitude_from_B(bcoeffs, mn)
        amplitude[pupil] += acoeffs
    if bcoeffs is None:
        amplitude = pupil
                
    phase += static_phase # extra static phase term (from a previous estimate, perhaps)
    return simulate_psf(amplitude, phase, wavelen, focal_length, coords1, coords2, to_focal_plane=to_focal_plane)

def get_sim_psfs(pupil, zbasis, bg, defocus_values, wavelen, f, pupil_coords, focal_coords, zcoeffs=None, pcoeffs=None, bcoeffs=None, static_phase=0, to_focal_plane=False):
    # Fresnel propagate from pupil to paraxial focus
    xp = get_array_module(pupil)
    incoh_psf_focus, Efocus, Epupil = simulate_from_coeffs(pupil, zbasis, wavelen, f, pupil_coords, focal_coords,
                                                           zcoeffs=zcoeffs, pcoeffs=pcoeffs, bcoeffs=bcoeffs,
                                                           static_phase=static_phase, to_focal_plane=to_focal_plane)

    # angular spectrum propagate to K defocus values
    Eks, Uks = propagate_by_angular_spectrum(Efocus, wavelen, defocus_values, pupil_coords[-1])
    psfs = (Eks * Eks.conj()).real
    psfs = xp.asarray(psfs) + bg
    return psfs, xp.asarray(Eks), xp.asarray(Uks), Epupil

#--------------INVERSE PROBLEM----------------

def obj_func(params, keys, ukeys, key_param_mapping, param_dict, meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, jac, return_all_the_things=False):
    '''
    Get the value of the objective function and jacobian for a set of parameters
    '''

    xp = get_array_module(meas_psfs)
    # copy the param_dict and update it with the fit parameters, as needed
    # here we also do the conversion from CPU to GPU memory if needed
    curdict = param_dict#deepcopy(param_dict)
    params = xp.asarray(params)
    #ukeys = np.unique(keys)
    for key, mapping in zip(ukeys, key_param_mapping):
        curdict[key] = [xp.asarray(params)[mapping], True]

    # simulate PSFs from parameters
    Ikhat, Ekhat, Ukhat, Ephat, Gkhat, gkhat, Hk, Hd, fkhat = get_Gkhat(pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, curdict)

    # just for consistency's sake
    Gk = meas_psfs - curdict['bg'][0][0]

    #----compute the objective function----
    lambda1 = arg_options.get('lambda1', 0)
    lambda2 = arg_options.get('lambda2', 0)
    mn = xp.count_nonzero(pupil)
    B = curdict['point_by_point_ampB'][0]
    fitamp = curdict['point_by_point_ampB'][1]
    if (lambda1 != 0) and fitamp:
        kappa1 = arg_options['kappa1']
        phi_1 = obj_phi1(B, pupil, kappa1, mn)
    else:
        phi_1 = 0
    # get Phi_2 and its jacobian
    if (lambda2 != 0) and fitamp:
        kappa2 = arg_options['kappa2']
        phi_2 = obj_phi2(B, pupil, kappa2, mn)
    else:
        phi_2 = 0
     
    # weighted sum of objective funcs
    obj = get_Phi(weighting, Gk, Gkhat) + lambda1 * phi_1 + lambda2 * phi_2

    if isinstance(obj, cp.core.ndarray):
        obj = cp.asnumpy(obj)
    
    # compute the Jacobian terms
    if jac:
        # ---first, the setup---

        # used by all jacobian terms
        dPhi_dGkhat = get_dPhi_dGkhat(weighting, Gkhat, Gk)
        gkdagger = get_gkdagger(dPhi_dGkhat)

        # only calculate if zk, zcoeffs, phase, or amp are optimized
        if np.any([curdict['zk'][1], curdict['zcoeffs'][1], curdict['point_by_point_phase'][1], curdict['point_by_point_ampB'][1]]):
            fkdagger = get_fkdagger(gkdagger, Hk, Hd, curdict['focal_plane_blur'][0][0]) 
            dPhi_dIk = get_dPhi_dIk(fkdagger)
            Ekdagger = get_Ekdagger(Ekhat, dPhi_dIk)
            Ukdagger = get_Ukdagger(Ekdagger)

        # only calculate if zcoeffs, phase, or amp are optimized
        if np.any([curdict['zcoeffs'][1], curdict['point_by_point_phase'][1], curdict['point_by_point_ampB'][1]]):
            Ufdagger = get_Ufdagger(Ukdagger, curdict['zk'][0], wavelen, pupil_coords[-1])
            Efdagger = get_Efdagger(Ufdagger)
            Epdagger = get_Epdagger(Efdagger)

        # ---now, compute the actual gradient terms---
        jaclist = []

        # x, y gradient
        if curdict['xk'][1] or curdict['yk'][1]:
            N = meas_psfs.shape[-1]
            gradsx = get_dPhi_dxk(gkdagger, gkhat, N)
            gradsy = get_dPhi_dyk(gkdagger, gkhat, N) 
            jaclist.append(gradsx)
            jaclist.append(gradsy)

        # z gradient
        if curdict['zk'][1]:
            gradsz = get_dPhi_dzk(Ukdagger, Ukhat, pupil_coords[-1], wavelen)
            jaclist.append(gradsz)

        # zernike gradient
        if curdict['zcoeffs'][1]: #Epdagger
            gradzern = get_dPhi_dzcoeff(Epdagger, Ephat, zbasis)
            jaclist.append(gradzern)

        # estimate point-by-point phase
        if curdict['point_by_point_phase'][1]:
            smoothing = arg_options.get('smoothing', None)
            #print(f'smoothing: {smoothing}')
            gradphase = get_dPhi_dphi(Epdagger, Ephat, smoothing=smoothing)[pupil]
            jaclist.append(gradphase)
            
        if curdict['point_by_point_ampB'][1]:
            
            phi_hat = static_phase + param_dict_to_phase(curdict, pupil, zbasis)
            amp_smoothing = arg_options.get('amp_smoothing', None)
            dPhi_dA = get_dPhi_dA(Epdagger, phi_hat, smoothing=amp_smoothing,)[pupil]
            # integrate dPhi/dB, dPhi1/dB, and dPhi2/dB (weighted sum of jacobians)
            dPhi_dB = get_amplitude_grad(dPhi_dA, curdict, meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options)
            jaclist.append(dPhi_dB)

        jac = xp.hstack(jaclist)

        if not isinstance(jac, np.ndarray):
            # convert back to np array for scipy.optimize compatibility
            jac = cp.asnumpy(jac)

        if return_all_the_things:
            return {
                'obj' : obj,
                'Ikhat' : Ikhat,
                'Ephat' : Ephat,
                'Ukhat' : Ukhat,
                'fkhat' : fkhat,
                'gkhat' : gkhat,
                'Gkhat' : Gkhat,
                'Gk' : Gk,
                'Hk' : Hk,
                'dPhi_dGkhat' : dPhi_dGkhat,
                'gkdagger' : gkdagger,
                'fkdagger' : fkdagger,
                'dPhi_dIk' : dPhi_dIk,
                'Ekdagger' : Ekdagger,
                'Ukdagger' : Ukdagger,
                'Ufdagger' : Ufdagger,
                'Efdagger' : Efdagger,
                'Epdagger' : Epdagger,
                'jac' : jac}
        return obj, jac
    else:
        return obj


def get_Gkhat(pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, curdict):
    # simulate PSFs from parameters
    xp = get_array_module(pupil)
    Ikhat, Ekhat, Ukhat, Ephat = get_sim_psfs(pupil, zbasis, 0, curdict['zk'][0], wavelen, f, pupil_coords, focal_coords,
                                                      zcoeffs=curdict['zcoeffs'][0],
                                                      bcoeffs=curdict['point_by_point_ampB'][0],
                                                      static_phase=static_phase,
                                                      pcoeffs=curdict['point_by_point_phase'][0],
                                                      to_focal_plane=True)
        
    # fkhat, gkhat, Gkhat
    shape = Ikhat[0].shape
    fkhat = fft2_shiftnorm(Ikhat, axes=(1,2), shift=False) # DFT of Ikhat
    pixel_coords = get_coords_fftshift(shape[0], xp=xp)
    Hk = xp.asarray([my_fourier_shift(yk, xk, shape, xp=xp, coords=pixel_coords) for (xk, yk) in zip(curdict['xk'][0], curdict['yk'][0])])
    Hd = get_Hd(Ikhat[0].shape[0], xp=xp, fftshift=True) #fftshift?
    gkhat = fkhat * Hk * Hd
    blur_sigma = curdict['focal_plane_blur'][0][0]
    if blur_sigma != 0:
        pass
        #gkhat *= fft2_shiftnorm(get_gauss(blur_sigma, shape, xp=xp) , norm=None, shift=True) #* shape[0]
    Gkhat = ifft2_shiftnorm(gkhat, axes=(1,2), shift=False).real # modeled PSF
    
    return Ikhat, Ekhat, Ukhat, Ephat, Gkhat, gkhat, Hk, Hd, fkhat

def get_Hd(N, xp=np, fftshift=False):
    #yy, xx = xp.indices((N,N)) - N/2.
    yy, xx, rr = get_coords_fftshift(N, xp=xp)
    #if fftshift:
    #    shift = xp.fft.fftshift # iffshift?
    #else:
    #    shift = lambda x: x
    return xp.sinc(yy/N)*xp.sinc(xx/N) #shift

def get_Phi(weighting, meas_psfs, sim_psfs):
    # objective func
    xp = get_array_module(meas_psfs)
    t1 = xp.sum(weighting*meas_psfs*sim_psfs, axis=(1,2))**2
    t2 = xp.sum(weighting*meas_psfs**2, axis=(1,2))
    t3 = xp.sum(weighting*sim_psfs**2, axis=(1,2))
    return 1 - 1./meas_psfs.shape[0] * xp.sum(t1/(t2*t3), axis=0) # should be >0 and =0 when meas_psfs = sim_psfs

def get_dPhi_dGkhat(W, Ghat, G):
    '''
    W = weighting
    Ghat = esimated psfs
    G = measured psfs
    '''
    xp = get_array_module(G)
    K = W.shape[0]
    
    WGhatG = xp.sum(W*Ghat*G, axis=(1,2))[:, xp.newaxis, xp.newaxis]
    WGhatsq = xp.sum(W*Ghat**2, axis=(1,2))[:, xp.newaxis, xp.newaxis]
    WGsq = xp.sum(W*G**2, axis=(1,2))[:, xp.newaxis, xp.newaxis]
    
    return 2./K * W * WGhatG / (WGsq * WGhatsq**2) * ( Ghat * WGhatG - G * WGhatsq)

def get_gkdagger(dPhi_dGkhat):
    return fft2_shiftnorm(dPhi_dGkhat, axes=(1,2), shift=False)

def get_fkdagger(gkdagger, shifts, Hd, blur_sigma):
    # maybe need to revist pixel and coordinate transfer functions
    #shifts = np.asarray([my_fourier_shift(-y,-x, shape) for (x,y) in zip(xk,yk)])
    xp = get_array_module(gkdagger)
    #Hd = get_Hd(shifts[0].shape[0], xp=xp, fftshift=True)
    if blur_sigma != 0:
        gauss = 1 #fft2_shiftnorm(get_gauss(blur_sigma, shifts[0].shape, xp=xp) , norm=None, shift=False).conj() #* shifts[0].shape[0]
    else:
        gauss = 1
    return gkdagger * shifts.conj() * Hd.conj() * gauss

def get_dPhi_dIk(fkdagger):
    return ifft2_shiftnorm(fkdagger, axes=(1,2), shift=False)# depends on fkdagger

def get_Ekdagger(Ek, dPhi_dIk):
    return Ek*dPhi_dIk*2 #depends on dPhi_dIk

def get_Ukdagger(Ekdagger):
    return fft2_shiftnorm(Ekdagger, axes=(1,2), shift=False)

def get_Ufdagger(Ukdagger, zkhat, wavelen, r_pupil):
    xp = get_array_module(Ukdagger)
    tf = get_free_space_tf(zkhat, wavelen, r_pupil)#np.asarray([get_free_space_tf(z, wavelen, r_pupil) for z in zkhat])
    return xp.sum( Ukdagger * tf.conj(), axis=0)

def get_Efdagger(Ufdagger):
    return ifft2_shiftnorm(Ufdagger, shift=False)

def get_Epdagger(Efdagger):
    return ifft2_shiftnorm(Efdagger, shift=False)

def get_dPhi_dzk(Ukdagger, Uk, r_pupil, wavelen):
    '''
    Gradient of metric wrt defocus values z_k
    '''
    xp = get_array_module(Ukdagger)
    pi = xp.pi
    #tf = ne.evaluate('2*pi * sqrt(1/wavelen**2 - r_pupil**2)') #2*np.pi* csqrt(1./wavelen**2 - rfreq)
    tf = 2*pi * xp.sqrt(1/wavelen**2 - r_pupil**2 + 0j)
    return xp.imag(xp.sum(tf*Ukdagger*Uk.conj(), axis=(1,2)))

def get_dPhi_dphi(Epdagger, Ep, smoothing=None):
    '''
    Gradient of metric wrt phase values
    '''
    out = (Epdagger*Ep.conj()).imag
    if smoothing is not None:
        out = gauss_convolve(out, smoothing, force_real=True)
        #out = gaussian_filter(out, smoothing, mode='constant')
    return out

def get_dPhi_dzcoeff(Epdagger, Ep, zmode):
    '''
    Gradient of metric wrt zernike phase coeff
    '''
    return ( np.sum(Epdagger*Ep.conj()*zmode, axis=(1,2)) ).imag

def get_dPhi_dxk(gkdagger, gkhat, N):
    '''
    Gradient of metric wrt x shift
    '''
    #idx = np.indices((N,N))[1] - N/2.
    xp = get_array_module(gkdagger)
    #idy, idx, r = get_coords(N, 1, xp)
    idy, idx, r = get_coords_fftshift(N, xp)
    return - xp.imag(xp.sum(2*xp.pi*idx/N * gkdagger*gkhat.conj(), axis=(1,2)))

def get_dPhi_dyk(gkdagger, gkhat, N):
    '''
    Gradient of metric wrt y shift
    '''
    xp = get_array_module(gkdagger)
    #idy, idx, r = get_coords(N, 1, xp)
    idy, idx, r = get_coords_fftshift(N, xp)
    return - xp.imag(xp.sum(2*xp.pi*idy/N * gkdagger*gkhat.conj(), axis=(1,2))) 
                 
def get_dPhi_dA(Epdagger, phi_hat, smoothing=None):
    out =  (Epdagger * cexp(-1*phi_hat)).real
    if smoothing is not None:
        #out = gaussian_filter(out, smoothing, mode='constant')
        out = gauss_convolve(out, smoothing, force_real=True)
    return out

def get_dPhi_dAzcoeff(Epdagger, phi_hat, zbasis):
    xp = get_array_module(Epdagger)
    return xp.real( xp.sum(zbasis * Epdagger * cexp(-1*phi_hat), axis=(1,2)) )

def amplitude_from_B(B, mn):
    #mn = np.count_nonzero(pupil)
    xp = get_array_module(B)
    return mn*xp.abs(B)/xp.sum(xp.abs(B))

def get_dPhi_dB(dPhi_dA, A, B, mn):
    xp = get_array_module(dPhi_dA)
    return xp.sign(B) / xp.sum(xp.abs(B)) * ( mn*dPhi_dA - xp.sum(dPhi_dA*A) )

def param_dict_to_phase(param_dict, pupil, zbasis):
    xp = get_array_module(pupil)
    phase = xp.zeros(pupil.shape)
    phase[pupil] = param_dict['point_by_point_phase'][0] # output point-by-point
    phase += xp.einsum('i,ijk->jk', param_dict['zcoeffs'][0], zbasis) #np.sum(param_dict['zcoeffs'][0][:,None,None]*zbasis,axis=0) # input zernikes
    return phase

def param_dict_to_amplitude(param_dict, pupil):
    xp = get_array_module(pupil)
    amplitude = xp.zeros(pupil.shape)
    amplitude[pupil] = amplitude_from_B(param_dict['point_by_point_ampB'][0], xp.count_nonzero(pupil)) # output point-by-point
    return amplitude

def get_amplitude_grad(dPhi_dA, param_dict, meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options):
    
    xp = get_array_module(dPhi_dA)
    # get weighting
    lambda1 = arg_options.get('lambda1', 0)
    lambda2 = arg_options.get('lambda2', 0)
    
    # get jacobian of phi_d
    mn = xp.count_nonzero(pupil)
    B = param_dict['point_by_point_ampB'][0]
    A = amplitude_from_B(B, mn)
    dPhi_dB = get_dPhi_dB(dPhi_dA, A, B, mn)
    
    Bsq = xp.zeros(pupil.shape)
    Bsq[pupil] = B
    Asq = amplitude_from_B(Bsq, mn)
    
    # get Phi_1 and its jacobian
    if lambda1 != 0:
        kappa1 = arg_options['kappa1']
        dPhi1_dB = get_dPhi1_dB(Asq, Bsq, kappa1, mn)[pupil]
    else:
        dPhi1_dB = 0
        
    # get Phi_2 and its jacobian
    if lambda2 != 0:
        kappa2 = arg_options['kappa2']
        dPhi2_dB = get_dPhi2_dB(Asq, Bsq, kappa2, mn)[pupil]
    else:
        dPhi2_dB = 0
        
    jac = dPhi_dB + lambda1 * dPhi1_dB + lambda2 * dPhi2_dB # weighted sum of jacobians
    return jac
        
def obj_phi1(B, pupil, kappa1, mn):
    #mn = np.count_nonzero(pupil) #pupil.shape[0] * pupil.shape[1]
    xp = get_array_module(B)
    Bsq = xp.zeros(pupil.shape)
    Bsq[pupil] = B
    A = amplitude_from_B(Bsq, mn)
    return get_Phi1(A, kappa1, mn)#, get_dPhi1_dB(A, Bsq, kappa1, mn)[pupil.astype(bool)]

def obj_phi2(B, pupil, kappa2, mn):
    #mn = np.count_nonzero(pupil) #pupil.shape[0] * pupil.shape[1]
    xp = get_array_module(B)
    Bsq = xp.zeros(pupil.shape)
    Bsq[pupil] = B
    A = amplitude_from_B(Bsq, mn)
    return get_Phi2(A, kappa2, mn)#, get_dPhi2_dB(A, Bsq, kappa2, mn)[pupil.astype(bool)]

def get_Phi1(A, kappa1, mn):
    xp = get_array_module(A)
    return 1./mn * xp.sum(Gamma(A, kappa1))

def get_Phi2(A, kappa2, mn):
    xp = get_array_module(A)
    shifts = [(0,1), (1,0), (1,1), (1,-1)] # these could be backwards
    Arolls = xp.asarray([np.roll(A, sh, axis=(0,1)) for sh in shifts])
    return 1./mn * xp.sum(Gamma(A - Arolls, kappa2))

def get_dPhi1_dA(A, kappa1, mn):
    return 1./mn * Gammaprime(A, kappa1)

def get_dPhi2_dA(A, kappa2, mn):
    xp = get_array_module(A)
    shifts = [(0,1), (1,0), (1,1), (1,-1)]
    Arolls = xp.asarray([xp.roll(A, sh, axis=(0,1)) for sh in shifts])
    Arollsneg = xp.asarray([xp.roll(A, (-sh[0], -sh[1]), axis=(0,1)) for sh in shifts])
    return 1./mn * xp.sum(Gammaprime(A - Arolls, kappa2) - Gammaprime(Arollsneg - A, kappa2), axis=0)

def get_dPhi1_dB(A, B, kappa1, mn):
    dPhi1_dA = get_dPhi1_dA(A, kappa1, mn)
    return get_dPhi_dB(dPhi1_dA, A, B, mn)

def get_dPhi2_dB(A, B, kappa2, mn):
    dPhi2_dA = get_dPhi2_dA(A, kappa2, mn)
    return get_dPhi_dB(dPhi2_dA, A, B, mn)

def Gamma(x, kappa):
    xp = get_array_module(x)
    absx = xp.abs(x)
    out = xp.zeros_like(x)
    out = 2/(3*kappa**2)*absx**2 - 8/(27*kappa**3)*absx**3 + 1./(27*kappa**4)*absx**4
    out[absx > 3*kappa] = 1
    return out

def Gammaprime(x, kappa):
    xp = get_array_module(x)
    absx = xp.abs(x)
    out = xp.zeros_like(x)
    out = xp.sign(x) * (4/(3*kappa**2)*absx - 8/(9*kappa**3)*absx**2 + 4./(27*kappa**4)*absx**3)
    out[absx > 3*kappa]  = 0
    return out

#--------------FITTING HELPERS----------------

DEFAULT_OPTIONS = {'gtol' : 1e-6, 'ftol' : 1e-6, 'maxcor' : 1000, 'maxls' : 100}

rough_smooth = 3
fine_smooth = 1
very_fine = None

lambda1 = 0.5
lambda2 = 0.125
kappa1 = 1
kappa2 = 1

DEFAULT_STEPS = [
    # lateral
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : True, 'yk' : True, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'opt_options' : DEFAULT_OPTIONS},
    # lateral + axial
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'opt_options' : DEFAULT_OPTIONS},
    # lateral + axial + zcoeffs
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : True, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'opt_options' : DEFAULT_OPTIONS},
    # phase: rough
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : rough_smooth}, 'opt_options' : DEFAULT_OPTIONS},
    # lateral + axial + phase (fine)
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : fine_smooth}, 'opt_options' : DEFAULT_OPTIONS},
    # amplitude: rough
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' :fine_smooth, 'amp_smoothing' : rough_smooth, 'lambda1' : lambda1, 'lambda2' : lambda2, 'kappa1' : kappa1, 'kappa2' : kappa2}, 'opt_options' : DEFAULT_OPTIONS},
    # amplitude: fine
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : fine_smooth, 'amp_smoothing' : fine_smooth, 'lambda1' : lambda1, 'lambda2' : lambda2, 'kappa1' : kappa1, 'kappa2' : kappa2}, 'opt_options' : DEFAULT_OPTIONS},
    # lateral + axial + phase (fine) + amplitude (fine)
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : fine_smooth, 'amp_smoothing' : fine_smooth, 'lambda1' : lambda1, 'lambda2' : lambda2, 'kappa1' : kappa1, 'kappa2' : kappa2}, 'opt_options' : DEFAULT_OPTIONS},
    # phase: pixel
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : very_fine, 'lambda1' : lambda1, 'lambda2' : lambda2, 'kappa1' : kappa1, 'kappa2' : kappa2}, 'opt_options' : DEFAULT_OPTIONS},
    # amplitude: pixel
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : very_fine, 'amp_smoothing' : very_fine, 'lambda1' : lambda1, 'lambda2' : lambda2, 'kappa1' : kappa1, 'kappa2' : kappa2}, 'opt_options' : DEFAULT_OPTIONS},
    # lateral + axial + phase (pixel) + amplitude (pixel)
    {'bg' : False, 'focal_plane_blur' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : very_fine, 'amp_smoothing' : very_fine, 'lambda1' : lambda1, 'lambda2' : lambda2, 'kappa1' : kappa1, 'kappa2' : kappa2}, 'opt_options' : DEFAULT_OPTIONS},
]

def estimate_phase_from_measured_psfs(meas_psfs, param_dict, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, weighting, arg_options={}, options={}, method='L-BFGS-B', jac=True):
    '''
    Perform a single step of the (normally multi-step) estimation routine
    '''
    # get initial guesses for parameters to optimize
    x0, keys = param_dict_to_list(param_dict)

    ukeys = np.unique(keys)
    key_param_mapping = [np.asarray(keys) == key for key in ukeys]
    
    init_obj = obj_func(deepcopy(x0), deepcopy(keys), ukeys, key_param_mapping, deepcopy(param_dict), meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, False)
    logger.info(f'Initial value of objective func: {init_obj}')
    
    # run the optimizer
    opt = minimize(obj_func, deepcopy(x0), args=(deepcopy(keys), ukeys, key_param_mapping, deepcopy(param_dict), meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, jac),
                    method=method,jac=jac, options=options)
    logger.info(f'Final value of objective func: {opt["fun"]}')
    
    #final_obj = obj_func([], [], deepcopy(param_dict), meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, False)
    #print(f'Final value of objective func (re-evaluated): {opt["fun"]}')

    return opt, update_param_dict_from_list(keys, opt['x'], param_dict)
        
def multi_step_fit(measured_psfs, pupil, z0vals, zbasis, wavelen, z0, weighting, pupil_coords, focal_coords, w_threshold=10, w_eta=1e-3, input_phase=None, input_amp=None, input_params=None, steps=DEFAULT_STEPS, options=DEFAULT_OPTIONS, method='L-BFGS-B', jac=True, xk=None, yk=None, focal_plane_blur=0, gpu=True):
    '''
    Perform a sequence of estimation steps, typically to walk toward the solution
    '''
    k = measured_psfs.shape[0]
    mn = np.count_nonzero(pupil)
    z = zbasis.shape[0]

    if xk is None:
        xk = np.zeros(k)
    if yk is None:
        yk = np.zeros(k)
    
    if input_amp is None:
        input_amp = np.ones(pupil.shape)
    init_amp = input_amp[pupil]

    if input_phase is None:
        input_phase = np.zeros(pupil.shape)
    init_phase = input_phase[pupil]
    
    # set up the fitter dict (all must be CPU / numpy)
    param_dict = OrderedDict({
    'bg' : [[get_median(measured_psfs),], False],
    'xk' : [xk, False], # initial guess and whether to fit
    'yk' : [yk, False],
    'zk' : [deepcopy(z0vals), False],
    'zcoeffs' : [np.zeros(z), False],
    'point_by_point_phase' : [init_phase, False],
    'point_by_point_ampB' : [init_amp, False], # amplitude = 1 everywhere
    'focal_plane_blur' : [np.asarray([focal_plane_blur,]), False]
    })


    # if GPU, convert arguments (everything but input parameters) to cupy
    if gpu is True:
        # convert from np to cp (CPU -> GPU)
        pupil = cp.asarray(pupil)
        zbasis = cp.asarray(zbasis)
        pupil_coords = cp.asarray(pupil_coords)
        focal_coords = cp.asarray(focal_coords)
        measured_psfs = cp.asarray(measured_psfs)
        weighting = cp.asarray(weighting)
        xp = cp
    else:
        xp = np
    
    curdict = deepcopy(param_dict)
    arg_options = {}
    # multi-step fitting
    for i, s in enumerate(steps):
        opt_options = deepcopy(options) # revert to defaults for optimization options
        logger.info(f'Step {i+1}/{len(steps)}: {s}')
        # update param_dict appropriately
        for key, val in s.items():
            if key.lower() == 'arg_options':
                arg_options = val
            elif key.lower() == 'opt_options':
                opt_options.update(val)
            else:
                curdict[key][1] = val # fit or don't fit
                if not val: # if not being fit, then update to cp/np as needed
                    curdict[key][0] = xp.asarray(curdict[key][0])
                else: # parameters that ARE being fit always need to be np arrays
                    if not isinstance(curdict[key][0], np.ndarray):
                        curdict[key][0] = cp.asnumpy(curdict[key][0])
            
        #print(opt_options)
        out, curdict = estimate_phase_from_measured_psfs(measured_psfs, curdict, pupil, zbasis, wavelen, z0, pupil_coords,
                                    focal_coords, 0, weighting, arg_options=arg_options, options=opt_options, method=method, jac=opt_options.pop('jac',jac))
        logger.info('Success? ' + str(out['success']))


    if gpu:
        # convert as necessary param_dict terms from GPU to CPU memory
        for key, values in curdict.items():
            if isinstance(values[0], cp.core.ndarray):
                curdict[key][0] = cp.asnumpy(values[0])
        pupil = cp.asnumpy(pupil)
        zbasis = cp.asnumpy(zbasis)
        pupil_coords = cp.asnumpy(pupil_coords)
        focal_coords = cp.asnumpy(focal_coords)

    # collect all the quantities to pass back
    total_phase = param_dict_to_phase(curdict, pupil, zbasis) + input_phase
    total_amp = param_dict_to_amplitude(curdict, pupil)
    sim_psf, Efocal, Epupil = simulate_psf(total_amp, total_phase, wavelen, z0, pupil_coords, focal_coords, to_focal_plane=True)

    #print(type(pupil), type(zbasis), type(pupil_coords[0]), type(focal_coords[0]))
    Ikhat, Ekhat, Ukhat, Ephat, Gkhat, gkhat, Hk, Hd, fkhat = get_Gkhat(pupil, zbasis, wavelen, z0, pupil_coords, focal_coords, 0, curdict)

    return out, curdict, total_phase, total_amp, Epupil, Efocal, sim_psf, Gkhat

def process_phase_retrieval(psfs, params, weights=None, input_phase=None, xk_in=None, yk_in=None, focal_plane_blur=0, gpu=True, method='L-BFGS-B', steps=DEFAULT_STEPS, options=DEFAULT_OPTIONS):
    '''
    This is a wrapper around multi_step_fit, to handle a lot of the data wrangling.
    '''
    # input params: fitting_region, zbasis, wavelen, f, pupil_coords, focal_coords, pupil_rescaled

    # weights defaults to 1/Poisson
    if weights is None:
        bg = np.median(psfs)
        weights = np.asarray([1./gauss_convolve(p + 5*bg, 5) for p in psfs]).astype(np.float64)

    # fft shifts
    fitting_region_shifted = np.fft.fftshift(params['fitting_region'])
    pupil_analytic_shifted = np.fft.fftshift(params['pupil_analytic'])
    pcoords_shifted = np.fft.fftshift(params['pupil_coords'], axes=(-2,-1))
    fcoords_shifted = np.fft.fftshift(params['focal_coords'], axes=(-2,-1))
    zbasis_shifted = np.fft.fftshift(params['zbasis'], axes=(-2,-1))

    psfs_shifted = np.fft.fftshift(psfs.astype(np.float64), axes=(-2,-1))
    weights_shifted = np.fft.fftshift(weights, axes=(-2,-1))

    N = psfs.shape[-1] # I'm assuming these are square images
    
    if input_phase is not None:
        input_phase = np.fft.fftshift(np.array(input_phase))
    
    coms_yx = [center_of_mass(p) for p in psfs]
    cenyx = [(N-1)/2., (N-1)/2.]
    if xk_in is None:
        out = [c[1] - cenyx[1] for c in coms_yx]
        xk_in = np.array( [c[1] - cenyx[1] for c in coms_yx] )
    if yk_in is None:
        yk_in = np.array([c[0] - cenyx[0] for c in coms_yx] )

    z0vals = params['zkvals']
    
    out_final, param_dict, est_phase, est_amp, est_Epupil, est_Efocal, est_psf, Gkhat = multi_step_fit(psfs_shifted, fitting_region_shifted, z0vals, zbasis_shifted,
        params['wavelen'], params['f'], weights_shifted, pcoords_shifted, fcoords_shifted, steps=steps, options=options, input_phase=input_phase,
        input_amp=pupil_analytic_shifted + fitting_region_shifted*1e-5, xk=xk_in, yk=yk_in, jac=True, focal_plane_blur=focal_plane_blur, method=method, gpu=gpu)
    
    # ifftshift and return
    return {
        'amp' : np.fft.ifftshift(est_amp),
        'phase' : np.fft.ifftshift(est_phase),
        'Epupil' : np.fft.ifftshift(est_Epupil),
        'Efocal' : np.fft.ifftshift(est_Efocal),
        'psf' : np.fft.ifftshift(est_psf),
        'obj_val' : out_final['fun'],
        'param_dict' : param_dict,
        'Gkhat' : np.fft.ifftshift(Gkhat, axes=(-2,-1))
    }

def _process_phase_retrieval_mpfriendly(params, input_phase, xk_in, yk_in, focal_plane_blur, gpu, method, steps, options, psfs):
    return process_phase_retrieval(psfs, params, input_phase=input_phase, xk_in=xk_in, yk_in=yk_in,
                                    focal_plane_blur=focal_plane_blur, gpu=gpu, method=method, steps=steps, options=options)

def multiprocess_phase_retrieval(allpsfs, params, input_phase=None, xk_in=None, yk_in=None, focal_plane_blur=0, gpu=True, method='L-BFGS-B', steps=DEFAULT_STEPS, options=DEFAULT_OPTIONS, processes=2):
    '''
    psfs, params, weights=None, input_phase=None, xk_in=None, yk_in=None, focal_plane_blur=0, gpu=True, method='L-BFGS-B', steps=DEFAULT_STEPS, options=DEFAULT_OPTIONS

    '''
    from functools import partial
    import multiprocessing as mp
    ctx = mp.get_context('spawn')

    mpfunc = partial(_process_phase_retrieval_mpfriendly, params, input_phase, xk_in, yk_in,
                     focal_plane_blur, gpu, method, steps, options)

    with ctx.Pool(processes=processes) as p:
        results = p.map(mpfunc, allpsfs)
    return results