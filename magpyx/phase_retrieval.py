import numpy as np
import poppy
import astropy.units as u
from copy import deepcopy
from itertools import product

import multiprocessing as mp
from functools import partial

from scipy.optimize import minimize
from numpy.lib.scimath import sqrt as csqrt
from scipy.optimize import minimize, leastsq
from scipy.ndimage.filters import gaussian_filter
from numpy.lib.scimath import sqrt as csqrt
from skimage.filters import threshold_otsu

from purepyindi import INDIClient

from poppy.zernike import arbitrary_basis
from scipy.ndimage import fourier_shift
from collections import OrderedDict
from astropy.io import fits
from time import sleep
from numpy.fft import fft2, ifft2
import pyfftw

from .utils import ImageStream, indi_send_and_wait

DEFAULT_STEPS = [
    {'bg' : True, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'opt_options' : {'jac' : False}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : None}},
    {'bg' : True, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'opt_options' : {'jac' : False}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : True, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : None}},
    {'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 2}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 0.75}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 2}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 0.75}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' :0.75, 'amp_smoothing' : 1.5}, 'opt_options' : {'ftol' : 1e-8}},
    #{'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : None, 'amp_smoothing' : None}, 'opt_options' : {'ftol' : 1e-8}},
    #{'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : None}, 'opt_options' : {'ftol' : 1e-8}}
]

amplitude_steps = [{'point_by_point_ampB' : True, 'arg_options' : {'lambda1' : 4, 'lambda2' : 0, 'kappa1' : .4, 'kappa2' : .4, 'amp_smoothing' : 1.25}},
                  {'point_by_point_ampB' : True, 'arg_options' : {'lambda1' : 0, 'lambda2' : 2, 'kappa1' : .4, 'kappa2' : .4}},
                  {'point_by_point_ampB' : True, 'arg_options' : {'lambda1' : 5, 'lambda2' : 1, 'kappa1' : .4, 'kappa2' : .4, 'amp_smoothing' : 1.25}}
                  ]
                  
STEPS_2 = [
    {'bg' : True, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'opt_options' : {'jac' : False}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 2.0}},
    {'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 2}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : False, 'arg_options' : {'smoothing' : 0.75}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : 0.75, 'amp_smoothing' : 2}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : False, 'yk' : False, 'zk' : False, 'zcoeffs' : False, 'point_by_point_phase' : False, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : 0.75, 'amp_smoothing' : 0.75}, 'opt_options' : {'ftol' : 1e-8}},
    {'bg' : False, 'xk' : True, 'yk' : True, 'zk' : True, 'zcoeffs' : False, 'point_by_point_phase' : True, 'point_by_point_ampB' : True, 'arg_options' : {'smoothing' : 0.75, 'amp_smoothing' : 0.75}, 'opt_options' : {'ftol' : 1e-8}}
]

DEFAULT_OPTIONS = {'gtol':1e-6, 'maxcor':1000, 'maxls' : 100, 'ftol' : 1e-8, 'eps' : 1e-12}

def multi_step_fit(measured_psfs, pupil, z0vals, zbasis, wavelen, z0, weighting, pupil_coords, focal_coords, w_threshold=10, w_eta=1e-3, input_phase=None, input_amp=None, input_params=None, steps=DEFAULT_STEPS, options=DEFAULT_OPTIONS, method='L-BFGS-B', jac=True, xk=None, yk=None):
    k = measured_psfs.shape[0]
    mn = np.count_nonzero(pupil)
    z = zbasis.shape[0]
    
    #print(k, mn, z)
    
    if input_phase is None:
        input_phase = np.zeros(pupil.shape)
        
    if xk is None:
        xk = np.zeros(k)
    if yk is None:
        yk = np.zeros(k)
    
    if input_amp is None:
        input_amp = np.zeros_like(pupil)
    init_amp = input_amp[pupil.astype(bool)]

    if input_phase is None:
        input_amp = np.zeros_like(pupil)
    init_phase = input_phase[pupil.astype(bool)]
    
    # set up the fitter dict
    param_dict = OrderedDict({
    'bg' : [[np.median(measured_psfs),], False],
    'xk' : [xk, False], # initial guess and whether to fit
    'yk' : [yk, False],
    'zk' : [deepcopy(z0vals), False],
    'zcoeffs' : [np.zeros(z), False],
    'point_by_point_phase' : [init_phase, False],
    'point_by_point_ampB' : [init_amp, False], # amplitude = 1 everywhere
    })
    
    # get pupil- and focal-plane coordinates 
    
    # wavelen and z0 -- can I abstract away from these?
    
    '''
    Other things:
    * can I get coordinates from pupil/psf fwhm fit?
    * can I get wavelen/z0 from the fit too? (should be a relationship between all these things)
    * do I need to do a background subtraction?
    * do I need to normalize/scale PSFs?
    * weighting should be computed here, maybe? (definitely not in estimate_phase_from_measured_psfs)
    '''
    
    curdict = deepcopy(param_dict)
    arg_options = {}
    # multi-step fitting
    for i, s in enumerate(steps):
        opt_options = deepcopy(options) # revert to defaults for optimization options
        print(f'Step {i+1}/{len(steps)}: {s}')
        # update param_dict appropriately
        for key, val in s.items():
            if key.lower() == 'arg_options':
                arg_options = val
            elif key.lower() == 'opt_options':
                opt_options.update(val)
            else:
                curdict[key][1] = val # fit or don't fit
            
        #print(opt_options)
        out, curdict = estimate_phase_from_measured_psfs(measured_psfs, curdict, pupil, zbasis, wavelen, z0, pupil_coords,
                                    focal_coords, 0, weighting, arg_options=arg_options, options=opt_options, method=method, jac=opt_options.pop('jac',jac))
        print(out['success'])

    return out, curdict, param_dict_to_phase(curdict, pupil, zbasis) + input_phase, param_dict_to_amplitude(curdict, pupil)

def get_magaox_pupil(npix, rotation=38.75, pupil_diam=6.5, support_width=0.01905, grid_size=6.5, sm=True, sm_scale=1, primary_scale=1):
    #pupil_diam = 6.5 #m
    secondary = 0.293 * pupil_diam * sm_scale
    primary = poppy.CircularAperture(radius=pupil_diam/2.*primary_scale)
    sec = poppy.AsymmetricSecondaryObscuration(secondary_radius=secondary/2.,
                                                 support_angle=(45, 135, 225, 315),
                                                 support_width=[support_width,]*4,
                                                 support_offset_y=[0.34,-0.34,-0.34,0.34],
                                                 rotation=rotation,
                                                 name='Complex secondary')
    opticslist = [primary,]
    if sm:
        opticslist.append(sec)
    pupil = poppy.CompoundAnalyticOptic( opticslist=opticslist, name='Magellan')
    sampled = pupil.sample(npix=npix, grid_size=grid_size)
    norm = np.sum(sampled)
    return pupil, sampled

def cexp(arr):
    return np.cos(arr) + 1j*np.sin(arr)

def fft2_shiftnorm(image, axes=None):
    #image = image.astype(np.float32)
    if axes is None:
        axes = (-2, -1)
    #return np.fft.fftshift(fft2(np.fft.ifftshift(image, axes=axes), axes=axes, norm='ortho'), axes=axes)
    t = pyfftw.builders.fft2(np.fft.ifftshift(image, axes=axes), axes=axes, threads=8, planner_effort='FFTW_ESTIMATE', norm='ortho')
    return np.fft.fftshift(t(),axes=axes)
    
def ifft2_shiftnorm(image, axes=None):
    #image = im.astype(np.float32)
    if axes is None:
        axes = (-2, -1)
    #return np.fft.fftshift(ifft2(np.fft.ifftshift(image, axes=axes), axes=axes, norm='ortho'), axes=axes)
    t = pyfftw.builders.ifft2(np.fft.ifftshift(image, axes=axes), axes=axes, threads=8, planner_effort='FFTW_ESTIMATE', norm='ortho')
    return np.fft.fftshift(t(), axes=axes)

def scale_and_noisify(psf, scale_factor, bg):
    psf_scaled = psf * scale_factor + bg
    poisson_noise = np.random.poisson(lam=np.sqrt(psf_scaled), size=psf_scaled.shape)
    return psf_scaled + poisson_noise

def fit_pupil_to_psf(camstream, nimages, sliceyx=None, padding=0, fwhm_guess=10):
    # measure an image
    image = np.mean(camstream.grab_many(nimages),axis=0).astype(float)
    if sliceyx is not None:
        image = image[sliceyx]
    meas_psf = image - np.median(image)
    meas_psf = shift_to_centroid(meas_psf)
    meas_psf = pad(meas_psf, padding)
    scale_factor, shifty, shiftx = fit_psf(meas_psf, get_magaox_pupil, fwhm_guess=fwhm_guess)[0]
    pupil, pupil_sampled = get_magaox_pupil(meas_psf.shape[0], grid_size=6.5*scale_factor)
    return pupil_sampled, scale_factor, shifty, shiftx  

def get_weighting(psf, threshold=2, eta=1e-8):
    # choose some kind of weighting per defocused psf -- high SNR pixels?
    bg = np.median(psf)
    snr = psf / (bg + eta)
    return (snr > threshold).astype(float)

def shift_to_centroid(im, shiftyx=None, order=3):
    if shiftyx is None:
        comyx = np.where(im == im.max())#com(im)
        cenyx = ((im.shape[0] - 1)/2., (im.shape[1] - 1)/2.)
        shiftyx = (cenyx[0]-comyx[0], cenyx[1]-comyx[1])
    median = np.median(im)
    return shift(im, shiftyx, order=1, cval=median)

def shift_via_fourier(image, xk, yk, force_real=False):
    out =  ifft2_shiftnorm(fft2_shiftnorm(image)*my_fourier_shift(xk, yk, image.shape))
    if force_real:
        return out.real
    else:
        return out

def pad(image, padlen):
    val = np.median(image)
    return np.pad(image, padlen, mode='constant', constant_values=val)

def fit_psf(meas_psf, pupil_func, fwhm_guess=10.):
    shape = meas_psf.shape[0]
    pupil_guess = fwhm_guess / 2. 
    return leastsq(psf_err, [pupil_guess, 0, 0], args=(pupil_func, shape, meas_psf), epsfcn=0.01)

def psf_err(params, pupil_func, sampling , meas_psf):
    scale_factor, ceny, cenx = params
    _, pupil = pupil_func(sampling, grid_size=6.5*scale_factor)
    
    sim_psf = shift_to_centroid(fraunhofer_simulate_psf(pupil, 1, 0, np.zeros((sampling, sampling)), add_noise=False), (ceny, cenx))

    #print(rms(sim_psf-meas_psf, np.ones_like(sim_psf).astype(bool)))
    return (normalize_psf(sim_psf) - normalize_psf(meas_psf)).flatten()

def normalize_psf(image):
    # normalize to unit energy
    return image / np.sum(image)

def get_free_space_tf(z, wavelen, rcoords):
    pi = np.pi
    #tf =  ne.evaluate('exp(1j*2*pi * z * sqrt(1/wavelen**2 - rcoords**2))')
    #tf = np.exp(1j*2*np.pi * z * csqrt(1/wavelen**2 - rcoords**2))
    tf = cexp(2*np.pi*z[:,None,None]*csqrt(1/wavelen**2 - rcoords**2))
    return tf
    
def lateral_shift_to_defocus_rms(delz, fnum):
    return -delz / (28 * (fnum)**2 )

def defocus_rms_to_lateral_shift(drms, fnum):
    return  -drms * 28 * (fnum)**2

def get_lens_phase(pupil, f, wavelen, r_coords):
    #pi = np.pi
    #return ne.evaluate('-2*pi/wavelen*(r_coords**2)/(2*f) * pupil')
    return -2*np.pi/wavelen*(r_coords**2)/(2*f) * pupil

def propagate_by_angular_spectrum(Efield, wavelen, z, rcoords):
    Uf = fft2_shiftnorm(Efield, axes=(-2,-1)) # angular spectrum of input field    
    Uk = Uf * get_free_space_tf(z, wavelen, rcoords) # propagate to K plane
    Ek = ifft2_shiftnorm(Uk, axes=(-2,-1)) # back to K E field
    return Ek, Uk

def fresnel_propagate(Efield, wavelen, rcoord1, rcoord2, z0, to_focal_plane=False):
    
    # E field plane coords
    k = 2*np.pi/wavelen
    pi = np.pi
    
    if to_focal_plane:
        fresnel_phasor = 1 # reduces to fourier transform of Efield
    else:
        fresnel_phasor =  cexp(k/(2*z0)*rcoord1**2)#np.exp(1j*k/(2*z0)*rcoord1**2) #ne.evaluate('exp(1j*k/(2*z0)*rcoord1**2)')
        
    #ne.evaluate('exp(1j*k*z0)/(1j*wavelen*z0)*exp(1j*pi*wavelen*z0*rcoord2)')
    #1/(wavelen*z0)
    #return np.exp(1j*k*z0)/(1j)*np.exp(1j*np.pi*wavelen*z0*rcoord2)
    return fft2_shiftnorm(Efield  * fresnel_phasor)

def get_sim_psfs(pupil, zbasis, bg, defocus_values, wavelen, f, pupil_coords, focal_coords, zcoeffs=None, pcoeffs=None, bcoeffs=None, static_phase=0, to_focal_plane=False):
    # Fresnel propagate from pupil to paraxial focus
    incoh_psf_focus, Efocus, Epupil = simulate_from_coeffs(pupil, zbasis, wavelen, f, pupil_coords, focal_coords,
                                                           zcoeffs=zcoeffs, pcoeffs=pcoeffs, bcoeffs=bcoeffs,
                                                           static_phase=static_phase, to_focal_plane=to_focal_plane)

    # angular spectrum propagate to K defocus values
    #psfs = []
    #Eks = []
    #Uks = []
    Eks, Uks = propagate_by_angular_spectrum(Efocus, wavelen, defocus_values, pupil_coords[-1])
    psfs = np.real(Eks * Eks.conj())
    #for zk in defocus_values:
    #    Ek, Uk = propagate_by_angular_spectrum(Efocus, wavelen, zk, pupil_coords[-1])
    #    psf = np.real(Ek * Ek.conj())
    #    psfs.append(psf)
    #    Eks.append(Ek)
    #    Uks.append(Uk)
    psfs = np.asarray(psfs) + bg
    return psfs, np.asarray(Eks), np.asarray(Uks), Epupil

def fraunhofer_simulate_psf(pupil, scale_factor, bg, phase, add_noise=True):
    pupil_field = pupil * cexp(phase)#np.exp(1j*phase)
    coh_psf = fft2_shiftnorm(pupil_field)
    psf = (coh_psf * coh_psf.conj()).real
    #norm = np.sum(psf)
    
    psf_scaled = psf * scale_factor + bg
    if add_noise:
        psf_out = scale_and_noisify(psf_scaled, 1., 0.)
    else:
        psf_out = psf_scaled
    return psf_out

def simulate_psf(pupil, phase, wavelen, focal_length, coords1, coords2, to_focal_plane=False):
    Epupil = pupil * cexp(phase)#np.exp(1j*phase) #ne.evaluate('pupil * exp(1j*phase)')
    norm = np.sqrt(np.sum(Epupil * Epupil.conj()).real)
    Epupil /= norm
    Efocal = fresnel_propagate(Epupil, wavelen, coords1[-1], coords2[-1], focal_length, to_focal_plane=to_focal_plane)
    return np.abs(Efocal)**2, Efocal, Epupil #incoherent and coherent psf

def simulate_from_coeffs(pupil, zbasis, wavelen, focal_length, coords1, coords2, zcoeffs=None, pcoeffs=None, bcoeffs=None, static_phase=0, to_focal_plane=False):
    phase = np.zeros(pupil.shape)
    amplitude = np.zeros(pupil.shape)
    if zcoeffs is not None: # zernike coeffs
        phase += np.einsum('i,ijk->jk', zcoeffs, zbasis) #np.sum(zcoeffs[:,None,None]*zbasis[:len(zcoeffs)],axis=0)
    if pcoeffs is not None: # pixel by pixel values
        phase[pupil.astype(bool)] += pcoeffs
    if bcoeffs is not None: # pixel by pixel amplitude values
        mn = np.count_nonzero(pupil)
        acoeffs = amplitude_from_B(bcoeffs, mn)
        amplitude[pupil.astype(bool)] += acoeffs
    if bcoeffs is None:
        amplitude = pupil
                
    phase += static_phase # extra static phase term (from a previous estimate, perhaps)
    return simulate_psf(amplitude, phase, wavelen, focal_length, coords1, coords2, to_focal_plane=to_focal_plane)

def my_fourier_shift(dely, delx, shape):
    cy, cx, r = get_coords(shape[0], 1) # assuming it's square
    return cexp(-2*np.pi*(dely*cy/shape[0] + delx*cx/shape[0]))#np.exp(-1j*2*np.pi*(dely*cy/shape[0] + delx*cx/shape[0]))

def get_coords(N, scale):
    cy, cx = (np.indices((N,N)) - N/2.) * scale
    r = np.sqrt(cy**2 + cx**2)
    return cy, cx, r

def get_Gkhat(pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, curdict):
    # simulate PSFs from parameters
    Ikhat, Ekhat, Ukhat, Ephat = get_sim_psfs(pupil, zbasis, 0, curdict['zk'][0], wavelen, f, pupil_coords, focal_coords,
                                                      zcoeffs=curdict['zcoeffs'][0],
                                                      bcoeffs=curdict['point_by_point_ampB'][0],
                                                      static_phase=static_phase,
                                                      pcoeffs=curdict['point_by_point_phase'][0],
                                                      to_focal_plane=True)
        
    # fkhat, gkhat, Gkhat
    shape = Ikhat[0].shape
    fkhat = fft2_shiftnorm(Ikhat, axes=(1,2)) # DFT of Ikhat
    Hk = np.asarray([my_fourier_shift(yk, xk, shape) for (xk, yk) in zip(curdict['xk'][0], curdict['yk'][0])])
    #Hd = get_Hd(Ikhat[0].shape[0])
    gkhat = fkhat * Hk #* Hd
    Gkhat = ifft2_shiftnorm(gkhat, axes=(1,2)).real # modeled PSF
    
    return Ikhat, Ekhat, Ukhat, Ephat, Gkhat, gkhat, Hk, fkhat

def obj_func(params, keys, param_dict, meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, jac, return_all_the_things=False):
    
    # copy the param_dict and update it with the fit parameters, as needed
    curdict = deepcopy(param_dict)
    ukeys = np.unique(keys)
    for key in ukeys:
        curdict[key] = [np.asarray(params)[np.asarray(keys) == key], True]

    # simulate PSFs from parameters
    Ikhat, Ekhat, Ukhat, Ephat, Gkhat, gkhat, Hk, fkhat = get_Gkhat(pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, curdict)
    '''    Ikhat, Ekhat, Ukhat, Ephat = get_sim_psfs(pupil, zbasis, 0, curdict['zk'][0], wavelen, f, pupil_coords, focal_coords,
                                                      zcoeffs=curdict['zcoeffs'][0],
                                                      bcoeffs=curdict['point_by_point_ampB'][0],
                                                      static_phase=static_phase,
                                                      pcoeffs=curdict['point_by_point_phase'][0],
                                                      to_focal_plane=True)
        
    # fkhat, gkhat, Gkhat
    shape = Ikhat[0].shape
    fkhat = fft2_shiftnorm(Ikhat, axes=(1,2)) # DFT of Ikhat
    Hk = np.asarray([my_fourier_shift(yk, xk, shape) for (xk, yk) in zip(curdict['xk'][0], curdict['yk'][0])])
    #Hd = get_Hd(Ikhat[0].shape[0])
    gkhat = fkhat * Hk #* Hd
    Gkhat = ifft2_shiftnorm(gkhat, axes=(1,2)).real # modeled PSF'''
    
    # just for consistency's sake
    Gk = meas_psfs - curdict['bg'][0][0]

    #----compute the objective function----
    lambda1 = arg_options.get('lambda1', 0)
    lambda2 = arg_options.get('lambda2', 0)
    mn = np.count_nonzero(pupil)
    B = curdict['point_by_point_ampB'][0]
    if lambda1 != 0:
        kappa1 = arg_options['kappa1']
        phi_1 = obj_phi1(B, pupil, kappa1, mn)
    else:
        phi_1 = 0
    # get Phi_2 and its jacobian
    if lambda2 != 0:
        kappa2 = arg_options['kappa2']
        phi_2 = obj_phi2(B, pupil, kappa2, mn)
    else:
        phi_2 = 0
     
    # weighted sum of objective funcs
    obj = get_Phi(weighting, Gk, Gkhat) + lambda1 * phi_1 + lambda2 * phi_2
    
    # compute the Jacobian terms
    if jac:
        # ---first, the setup---

        dPhi_dGkhat = get_dPhi_dGkhat(weighting, Gkhat, Gk)
        gkdagger = get_gkdagger(dPhi_dGkhat)
        fkdagger = get_fkdagger(gkdagger, Hk) 
        dPhi_dIk = get_dPhi_dIk(fkdagger)
        Ekdagger = get_Ekdagger(Ekhat, dPhi_dIk)
        Ukdagger = get_Ukdagger(Ekdagger)
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
            gradzern = get_dPhi_dzcoeff(Epdagger, Ephat, zbasis) # not quite working yet
            jaclist.append(gradzern)

        # estimate point-by-point phase
        if curdict['point_by_point_phase'][1]:
            smoothing = arg_options.get('smoothing', None)
            #print(f'smoothing: {smoothing}')
            gradphase = get_dPhi_dphi(Epdagger, Ephat, smoothing=smoothing)[pupil.astype(bool)]
            jaclist.append(gradphase)
            
        if curdict['point_by_point_ampB'][1]:
            
            phi_hat = static_phase + param_dict_to_phase(curdict, pupil, zbasis)
            amp_smoothing = arg_options.get('amp_smoothing', None)
            dPhi_dA = get_dPhi_dA(Epdagger, phi_hat, smoothing=amp_smoothing)[pupil.astype(bool)]
            '''
            N, M = meas_psfs[0].shape #pupil.shape
            mn = np.count_nonzero(pupil)
            B = curdict['point_by_point_ampB'][0]
            A = amplitude_from_B(B, mn)
            dPhi_dB = get_dPhi_dB(dPhi_dA, A, B, mn)
            '''
            # integrate dPhi/dB, dPhi1/dB, and dPhi2/dB (weighted sum of jacobians)
            dPhi_dB = get_amplitude_grad(dPhi_dA, curdict, meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options)
            jaclist.append(dPhi_dB)

        jac = np.hstack(jaclist)

        if return_all_the_things:
            return {
                'obj' : obj,
                'Ikhat' : Ikhat,
                'Ephat' : Ephat,
                'gkhat' : gkhat,
                'Gkhat' : Gkhat,
                'Gk' : Gk,
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
    
def estimate_phase_from_measured_psfs(meas_psfs, param_dict, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, weighting, arg_options={}, options={}, method='L-BFGS-B', jac=True):
    '''    
    Probably need to bg-subtract before passing in meas_psfs
    '''

    # adopt a weighting
    #weighting = np.asarray([get_weighting(p, threshold=0., eta=1e-3) for p in meas_psfs])
    
    # get initial guesses for parameters to optimize
    x0, keys = param_dict_to_list(param_dict)
    
    init_obj = obj_func(deepcopy(x0), deepcopy(keys), deepcopy(param_dict), meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, False)
    print(f'Initial value of objective func: {init_obj}')
    
    # run the optimizer
    opt = minimize(obj_func, deepcopy(x0), args=(deepcopy(keys), deepcopy(param_dict), meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, jac),
                    method=method,jac=jac, options=options)
    print(f'Final value of objective func: {opt["fun"]}')
    
    #final_obj = obj_func([], [], deepcopy(param_dict), meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options, False)
    #print(f'Final value of objective func (re-evaluated): {opt["fun"]}')

    return opt, update_param_dict_from_list(keys, opt['x'], param_dict)

def param_dict_to_list(param_dict):
    '''
    This is intended to give you a mapping of parameters to fit
    and their parameter type.
    '''
    to_fit = []
    keys_to_fit = []
    for key, val in param_dict.items():
        if val[-1] is True:
            to_fit.extend(deepcopy(list(val[0])))
            keys_to_fit.extend([key,]*len(val[0]))
    return to_fit, keys_to_fit
    
def update_param_dict_from_list(paramlist, x0, old_param_dict):
    updated_dict = deepcopy(old_param_dict)
    
    ukeys = np.unique(paramlist)
    for key in ukeys:
        updated_dict[key] = [np.asarray(x0, dtype=np.float64)[np.array(paramlist) == key], True]
    return updated_dict
        
def get_Hd(N):
    yy, xx = np.indices((N,N)) - N/2.
    return np.sinc(yy/N)*np.sinc(xx/N)

def get_Phi(weighting, meas_psfs, sim_psfs):
    # objective func
    t1 = np.sum(weighting*meas_psfs*sim_psfs, axis=(1,2))**2
    t2 = np.sum(weighting*meas_psfs**2, axis=(1,2))
    t3 = np.sum(weighting*sim_psfs**2, axis=(1,2))
    return 1 - 1./meas_psfs.shape[0] * np.sum(t1/(t2*t3), axis=0) # should be >0 and =0 when meas_psfs = sim_psfs

def get_dPhi_dGkhat(W, Ghat, G):
    '''
    W = weighting
    Ghat = esimated psfs
    G = measured psfs
    '''
    K = W.shape[0]
    
    WGhatG = np.sum(W*Ghat*G, axis=(1,2))[:, np.newaxis, np.newaxis]
    WGhatsq = np.sum(W*Ghat**2, axis=(1,2))[:, np.newaxis, np.newaxis]
    WGsq = np.sum(W*G**2, axis=(1,2))[:, np.newaxis, np.newaxis]
    
    return 2./K * W * WGhatG / (WGsq * WGhatsq**2) * ( Ghat * WGhatG - G * WGhatsq)

def get_gkdagger(dPhi_dGkhat):
    return fft2_shiftnorm(dPhi_dGkhat, axes=(1,2))

def get_fkdagger(gkdagger, shifts):
    # maybe need to revist pixel and coordinate transfer functions
    #shifts = np.asarray([my_fourier_shift(-y,-x, shape) for (x,y) in zip(xk,yk)])
    #Hd = get_Hd(shifts[0].shape[0])
    return gkdagger * shifts.conj() #* Hd.conj()

def get_dPhi_dIk(fkdagger):
    return ifft2_shiftnorm(fkdagger, axes=(1,2))# depends on fkdagger

def get_Ekdagger(Ek, dPhi_dIk):
    return Ek*dPhi_dIk*2 #depends on dPhi_dIk

def get_Ukdagger(Ekdagger):
    return fft2_shiftnorm(Ekdagger, axes=(1,2))

def get_Ufdagger(Ukdagger, zkhat, wavelen, r_pupil):
    tf = get_free_space_tf(zkhat, wavelen, r_pupil)#np.asarray([get_free_space_tf(z, wavelen, r_pupil) for z in zkhat])
    return np.sum( Ukdagger * tf.conj(), axis=0)

def get_Efdagger(Ufdagger):
    return ifft2_shiftnorm(Ufdagger)

def get_Epdagger(Efdagger):
    return ifft2_shiftnorm(Efdagger)

def get_dPhi_dzk(Ukdagger, Uk, r_pupil, wavelen):
    '''
    Gradient of metric wrt defocus values z_k
    '''
    pi = np.pi
    #tf = ne.evaluate('2*pi * sqrt(1/wavelen**2 - r_pupil**2)') #2*np.pi* csqrt(1./wavelen**2 - rfreq)
    tf = 2*pi * csqrt(1/wavelen**2 - r_pupil**2)
    return np.imag(np.sum(tf*Ukdagger*Uk.conj(), axis=(1,2)))

def get_dPhi_dphi(Epdagger, Ep, smoothing=None):
    '''
    Gradient of metric wrt phase values
    '''
    out = np.imag(Epdagger*Ep.conj())
    if smoothing is not None:
        out = gaussian_filter(out, smoothing, mode='constant')
    return out

def get_dPhi_dzcoeff(Epdagger, Ep, zmode):
    '''
    Gradient of metric wrt zernike phase coeff
    '''
    return np.imag( np.sum(Epdagger*Ep.conj()*zmode, axis=(1,2)) )
    
def get_dPhi_dA(Epdagger, phihat):
    '''
    Gradient of metric wrt amplitude values
    '''
    return np.real(Epdagger*cexp(-1*phihat))

def get_dPhi_dxk(gkdagger, gkhat, N):
    '''
    Gradient of metric wrt x shift
    '''
    #idx = np.indices((N,N))[1] - N/2.
    idy, idx, r = get_coords(N, 1)
    return - np.imag(np.sum(2*np.pi*idx/N * gkdagger*gkhat.conj(), axis=(1,2)))

def get_dPhi_dyk(gkdagger, gkhat, N):
    '''
    Gradient of metric wrt y shift
    '''
    idy, idx, r = get_coords(N, 1)
    return - np.imag(np.sum(2*np.pi*idy/N * gkdagger*gkhat.conj(), axis=(1,2))) 
                 
def get_dPhi_dA(Epdagger, phi_hat, smoothing=None):
    out =  np.real(Epdagger * cexp(-1*phi_hat))
    if smoothing is not None:
        out = gaussian_filter(out, smoothing, mode='constant')
    return out

def get_dPhi_dAzcoeff(Epdagger, phi_hat, zbasis):
    return np.real( np.sum(zbasis * Epdagger * cexp(-1*phi_hat), axis=(1,2)) )

def amplitude_from_B(B, mn):
    #mn = np.count_nonzero(pupil)
    return mn*np.abs(B)/np.sum(np.abs(B))

def get_dPhi_dB(dPhi_dA, A, B, mn):
    return np.sign(B) / np.sum(np.abs(B)) * ( mn*dPhi_dA - np.sum(dPhi_dA*A) )

def param_dict_to_phase(param_dict, pupil, zbasis):
    phase = np.zeros(pupil.shape)
    phase[pupil.astype(bool)] = param_dict['point_by_point_phase'][0] # output point-by-point
    phase += np.einsum('i,ijk->jk', param_dict['zcoeffs'][0], zbasis) #np.sum(param_dict['zcoeffs'][0][:,None,None]*zbasis,axis=0) # input zernikes
    return phase

def param_dict_to_amplitude(param_dict, pupil):
    amplitude = np.zeros(pupil.shape)
    amplitude[pupil.astype(bool)] = amplitude_from_B(param_dict['point_by_point_ampB'][0], np.count_nonzero(pupil)) # output point-by-point
    return amplitude

#--------amplitude metrics and gradients-------

'''def optimize_amplitude(B, pupil, mn, steps=DEFAULT_AMPLITUDE_STEPS, kappa1=1, kappa2=1, method='L-BFGS-B', jac=True, options=DEFAULT_OPTIONS):
    curB = deepcopy(B)
    for s in steps:
        opt = minimize(amplitude_obj, curB, args=(pupil, kappa1, kappa2, mn),method=method,jac=jac, options=options)
        print(opt)
        curB = opt['x'] 
    mn = np.count_nonzero(pupil) #pupil.shape[0] * pupil.shape[1]
    return curB, amplitude_from_B(curB, mn), opt'''

def get_amplitude_grad(dPhi_dA, param_dict, meas_psfs, weighting, pupil, zbasis, wavelen, f, pupil_coords, focal_coords, static_phase, arg_options):
    
    # get weighting
    lambda1 = arg_options.get('lambda1', 0)
    lambda2 = arg_options.get('lambda2', 0)
    
    # get jacobian of phi_d
    mn = np.count_nonzero(pupil)
    B = param_dict['point_by_point_ampB'][0]
    A = amplitude_from_B(B, mn)
    dPhi_dB = get_dPhi_dB(dPhi_dA, A, B, mn)
    
    Bsq = np.zeros(pupil.shape)
    Bsq[pupil.astype(bool)] = B
    Asq = amplitude_from_B(Bsq, mn)
    
    # get Phi_1 and its jacobian
    if lambda1 != 0:
        kappa1 = arg_options['kappa1']
        dPhi1_dB = get_dPhi1_dB(Asq, Bsq, kappa1, mn)[pupil.astype(bool)]
    else:
        dPhi1_dB = 0
        
    # get Phi_2 and its jacobian
    if lambda2 != 0:
        kappa2 = arg_options['kappa2']
        dPhi2_dB = get_dPhi2_dB(Asq, Bsq, kappa2, mn)[pupil.astype(bool)]
    else:
        dPhi2_dB = 0
        
    jac = dPhi_dB + lambda1 * dPhi1_dB + lambda2 * dPhi2_dB # weighted sum of jacobians
    return jac
        
def obj_phi1(B, pupil, kappa1, mn):
    #mn = np.count_nonzero(pupil) #pupil.shape[0] * pupil.shape[1]
    Bsq = np.zeros(pupil.shape)
    Bsq[pupil.astype(bool)] = B
    A = amplitude_from_B(Bsq, mn)
    return get_Phi1(A, kappa1, mn)#, get_dPhi1_dB(A, Bsq, kappa1, mn)[pupil.astype(bool)]

def obj_phi2(B, pupil, kappa2, mn):
    #mn = np.count_nonzero(pupil) #pupil.shape[0] * pupil.shape[1]
    Bsq = np.zeros(pupil.shape)
    Bsq[pupil.astype(bool)] = B
    A = amplitude_from_B(Bsq, mn)
    return get_Phi2(A, kappa2, mn)#, get_dPhi2_dB(A, Bsq, kappa2, mn)[pupil.astype(bool)]

def get_Phi1(A, kappa1, mn):
    return 1./mn * np.sum(Gamma(A, kappa1))

def get_Phi2(A, kappa2, mn):
    shifts = [(0,1), (1,0), (1,1), (1,-1)] # these could be backwards
    Arolls = np.asarray([np.roll(A, sh, axis=(0,1)) for sh in shifts])
    return 1./mn * np.sum(Gamma(A - Arolls, kappa2))

def get_dPhi1_dA(A, kappa1, mn):
    return 1./mn * Gammaprime(A, kappa1)

def get_dPhi2_dA(A, kappa2, mn):
    shifts = [(0,1), (1,0), (1,1), (1,-1)]
    Arolls = np.asarray([np.roll(A, sh, axis=(0,1)) for sh in shifts])
    Arollsneg = np.asarray([np.roll(A, (-sh[0], -sh[1]), axis=(0,1)) for sh in shifts])
    return 1./mn * np.sum(Gammaprime(A - Arolls, kappa2) - Gammaprime(Arollsneg - A, kappa2), axis=0)

def get_dPhi1_dB(A, B, kappa1, mn):
    dPhi1_dA = get_dPhi1_dA(A, kappa1, mn)
    return get_dPhi_dB(dPhi1_dA, A, B, mn)

def get_dPhi2_dB(A, B, kappa2, mn):
    dPhi2_dA = get_dPhi2_dA(A, kappa2, mn)
    return get_dPhi_dB(dPhi2_dA, A, B, mn)

def Gamma(x, kappa):
    absx = np.abs(x)
    out = np.zeros_like(x)
    out = 2/(3*kappa**2)*absx**2 - 8/(27*kappa**3)*absx**3 + 1./(27*kappa**4)*absx**4
    out[absx > 3*kappa] = 1
    return out

def Gammaprime(x, kappa):
    absx = np.abs(x)
    out = np.zeros_like(x)
    out = np.sign(x) * (4/(3*kappa**2)*absx - 8/(9*kappa**3)*absx**2 + 4./(27*kappa**4)*absx**3)
    out[absx > 3*kappa]  = 0
    return out

def command_stage(client, indi_triplet, value):
    command_dict = {indi_triplet : value}
    indi_send_and_wait(client, command_dict, tol=1e-2, wait_for_properties=True, timeout = 60)

def measure_defocused_psfs(camstream, defocus_pos, focus_pos, nimages):
    #dmstream.write(np.zeros((32,32)).astype(dmstream.buffer.dtype))
    images = []
    # positive values
    camstream.semflush(camstream.semindex)
    for pos in defocus_pos:
        #dmstream.write(cmd.astype(dmstream.buffer.dtype))
        command_stage(client, 'stagesci2.position.target', pos)
        camstream.semwait(camstream.semindex)
        sleep(0.1)
        images.append(np.mean(camstream.grab_many(nimages), axis=0))
    #dmstream.write(np.zeros((32,32)).astype(dmstream.buffer.dtype))
    command_stage(client, 'stagesci2.position.target', focus_pos)
    return images