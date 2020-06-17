'''
Dump from notebook.

To do:
* Remove unecessary imports / find missing imports
* Clean up functions
* Make functions for closed loop / end-to-end operations
* Generalize for ALPAO/BMC/DM of arbitrary shape (I guess)
* Add option for using DM model instead of measured IFs
* Add option for modal approach instead of IFs?
* Add command-line hooks of the form:
>> fnf camsci1 dmncpc --if_cube "/opt/MagAOX/etc/mygreatIFs.fits" --filter 1.4
>> fnf_calib camsci1 dmncpc --nrepeats 5 --sdflkj

* Test by getting working on the camscis with the NCPC DM?
'''

import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.ndimage.measurements import center_of_mass as com
from scipy.ndimage.interpolation import shift
from skimage.restoration import unwrap_phase

from magpyx.utils import ImageStream
import purepyindi as indi
from astropy.io import fits
import poppy
from poppy import zernike
from time import sleep
import dm_model as model
from scipy.linalg import hadamard
from copy import deepcopy
from scipy.optimize import leastsq
from skimage import draw

def pseudoinverse_svd(matrix, abs_threshold=None, rel_threshold=None, n_threshold=None):
    '''
    Compute the pseudo-inverse of a matrix via an SVD and some threshold.
    
    Only one type of threshold should be specified.
    
    Parameters:
        matrix: nd array
            matrix to invert
        abs_threshold: float
            Absolute value of sing vals to threshold
        rel_threshold: float
            Threshold sing vals < rel_threshold * max(sing vals)
        n_threshold : int
            Threshold beyond the first n_threshold singular values
        
    Returns:
        pseudo-inverse : nd array
            pseduo-inverse of input matrix
        threshold : float
            The absolute threshold computed
        U, s, Vh: nd arrays
            SVD of the input matrix
    '''
    from scipy import linalg

    
    if np.count_nonzero([abs_threshold is not None,
                         rel_threshold is not None,
                         n_threshold is not None]) > 1:
        raise ValueError('You must specify only one of [abs_threshold, rel_threshold, n_threshold]!')
        
    # take the SVD
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        
    #threshold
    if abs_threshold is not None:
        threshold = abs_threshold
        reject = s <= threshold
    elif rel_threshold is not None:
        threshold = s.max() * rel_threshold
        reject = s <= threshold
    elif n_threshold is not None:
        reject = np.arange(len(s)) > n_threshold
        threshold = s[n_threshold]
    else:
        threshold = 1e-16
        reject = s <= threshold
    
    sinv = np.diag(1./s).copy() # compute the inverse (this could create NaNs)
    sinv[reject] = 0. #remove elements that don't meet the threshold
    
    # just to be safe, remove any NaNs or infs
    sinv[np.isnan(sinv)] = 0.
    sinv[np.isinf(sinv)] = 0.
    
    # compute the pseudo-inverse: Vh.T s^-1 U_dagger (hermitian conjugate)
    return np.dot(Vh.T, np.dot(sinv, U.T.conj())), threshold, U, s, Vh


def get_even_odd_fft(im):
    # fft and use fourier properties to return the even/odd components
    fft = np.fft.fft2(im)
    even = np.fft.ifft2(fft.real).real
    odd = np.fft.ifft2(fft.imag*1j).real
    return even, odd

def get_y(p_odd, a , eta):
    return a * p_odd / (2*(a*a.conj()).real + eta)

def get_magv(p_even, S, a, y):
    magv = np.sqrt( np.abs(p_even - (S*(a*a.conj()).real + (y*y.conj()).real)) )
    return magv

def get_sgnv(p2_even, p1_even, vd, yd, y):
    sgnv = np.sign( (p2_even - p1_even - ( (vd*vd.conj()).real + (yd*yd.conj()).real + 2*y*yd)) / (2*vd))
    return sgnv

def get_a(pupil):
    coh_psf = fft2_shiftnorm(pupil)
    return coh_psf

def get_phase_estimate(sgnv, magv, y):
    return ifft2_shiftnorm(sgnv*magv - 1j*y)

def shift_to_centroid(im, order=3):
    comyx = np.where(im == im.max())#com(im)
    cenyx = ((im.shape[0] - 1)/2., (im.shape[1] - 1)/2.)
    shiftyx = (cenyx[0]-comyx[0], cenyx[1]-comyx[1])
    median = np.median(im)
    return shift(im, shiftyx, order=1, cval=median)

def fft2_shiftnorm(image):
    norm = np.sqrt(image.shape[0] * image.shape[1])
    return np.fft.fftshift(np.fft.fft2(image)) / norm
    
def ifft2_shiftnorm(image):
    norm = np.sqrt(image.shape[0] * image.shape[1])
    return np.fft.ifft2(np.fft.ifftshift(image)) * norm

def pad(image, padlen):
    val = np.median(image)
    return np.pad(image, padlen, mode='constant', constant_values=val)

def normalize_psf(image):
    # normalize to unit energy
    return image / np.sum(image)

def scale_and_noisify(psf, scale_factor, bg):
    psf_scaled = psf * scale_factor + bg
    poisson_noise = np.random.poisson(lam=np.sqrt(psf_scaled), size=psf_scaled.shape)
    return psf_scaled + poisson_noise

def window_image(image, fraction=1./np.sqrt(2), normalize=False):
    w1 = np.hanning(image.shape[0])
    w2 = np.hanning(image.shape[1])
    window = np.outer(w1, w2)
    #window = han2d(image.shape, fraction=fraction, normalize=normalize)
    return image * window

def han2d(shape, fraction=1./np.sqrt(2), normalize=False):
    '''
    Radial Hanning window scaled to a fraction 
    of the array size.
    
    Fraction = 1. for circumscribed circle
    Fraction = 1/sqrt(2) for inscribed circle (default)
    '''
    #return np.sqrt(np.outer(np.hanning(shape[0]), np.hanning(shape[0])))

    # get radial distance from center
    radial = get_radial_dist(shape)

    # scale radial distances
    rmax = radial.max() * fraction
    scaled = (1 - radial / rmax) * np.pi/2.
    window = np.sin(scaled)**2
    window[radial > fraction * radial.max()] = 0.
    return window

def get_radial_dist(shape, scaleyx=(1.0, 1.0)):
    '''
    Compute the radial separation of each pixel
    from the center of a 2D array, and optionally 
    scale in x and y.
    '''
    indices = np.indices(shape)
    cenyx = ( (shape[0] - 1) / 2., (shape[1] - 1)  / 2.)
    radial = np.sqrt( (scaleyx[0]*(indices[0] - cenyx[0]))**2 + (scaleyx[1]*(indices[1] - cenyx[1]))**2 )
    return radial

def rms(image, mask):
    mean = np.mean(image[mask])
    return np.sqrt(np.mean((image[mask]-mean)**2))

def simulate_psf(pupil, scale_factor, bg, phase, add_noise=True):
    pupil_field = pupil * np.exp(1j*phase)
    coh_psf = fft2_shiftnorm(pupil_field)
    psf = (coh_psf * coh_psf.conj()).real
    #norm = np.sum(psf)
    
    psf_scaled = psf * scale_factor + bg
    if add_noise:
        psf_out = scale_and_noisify(psf_scaled, 1., 0.)
    else:
        psf_out = psf_scaled
    return psf_out

def get_magaox_pupil(npix, rotation=38.75, grid_size=6.5):
    pupil_diam = 6.5 #m
    secondary = 0.293 * pupil_diam
    primary = poppy.CircularAperture(radius=pupil_diam/2.)
    sec = poppy.AsymmetricSecondaryObscuration(secondary_radius=secondary/2.,
                                                 #support_angle=(45, 135, 225, 315),
                                                 #support_width=[0.01905,]*4,
                                                 #support_offset_y=[0, -0.34, 0.34, 0],
                                                 rotation=rotation,
                                                 name='Complex secondary')
    pupil = poppy.CompoundAnalyticOptic( opticslist=[primary], name='Magellan')
    sampled = pupil.sample(npix=npix, grid_size=grid_size)
    norm = np.sum(sampled)
    return sampled

def get_magaox_pupil_sm(npix, rotation=38.75, grid_size=6.5):
    pupil_diam = 6.5 #m
    secondary = 0.293 * pupil_diam
    primary = poppy.CircularAperture(radius=pupil_diam/2.)
    sec = poppy.AsymmetricSecondaryObscuration(secondary_radius=secondary/2.,
                                                 #support_angle=(45, 135, 225, 315),
                                                 #support_width=[0.01905,]*4,
                                                 #support_offset_y=[0, -0.34, 0.34, 0],
                                                 rotation=rotation,
                                                 name='Complex secondary')
    pupil = poppy.CompoundAnalyticOptic( opticslist=[primary, sec], name='Magellan')
    sampled = pupil.sample(npix=npix, grid_size=grid_size)
    norm = np.sum(sampled)
    return sampled

def ff_estimate_phase(psf1, psf2, diff_phase, eta=10000, a=None, pupil=None, S=None):
    
    if a is None:
        a = get_a(pupil)
    if S is None:
        S = 0.8 #estimate_strehl
        
    p1_even, p1_odd = get_even_odd_fft(psf1)
    if psf2 is not None:
        p2_even, p2_odd = get_even_odd_fft(psf2)

    y = get_y(p1_odd, a, eta)
    magv = get_magv(p1_even, S, a, y)
    
    if psf2 is not None:
        pd_even, pd_odd = get_even_odd_fft(diff_phase)
        yd = fft2_shiftnorm(pd_odd*pupil)*1j 
        vd = fft2_shiftnorm(pd_even*pupil)
        sgnv= get_sgnv(p2_even, p1_even, vd, yd, y)
    else:
        sgnv = np.sign(a)

    phase_est = get_phase_estimate(sgnv, magv, y).real
    phase_est -= np.mean(phase_est[pupil.astype(bool)])
    phase_est *= pupil.astype(bool)
    return phase_est

def fit_pupil_to_psf(image, pupil_func, fwhm_guess=10):
    return fit_psf(image, pupil_func, fwhm_guess=fwhm_guess)
    
def fit_psf(meas_psf, pupil_func, fwhm_guess=10.):
    
    shape = meas_psf.shape[0]
    pupil_guess = fwhm_guess / 2. 
    return leastsq(psf_err, [pupil_guess,], args=(pupil_func, shape, meas_psf), epsfcn=0.1)

def psf_err(params, pupil_func, sampling , meas_psf):
    scale_factor = params[0]
    pupil = pupil_func(sampling, grid_size=6.5*scale_factor)
    
    sim_psf = simulate_psf(pupil, 1, 0, np.zeros((sampling, sampling)), add_noise=False)

    #print(rms(sim_psf-meas_psf, np.ones_like(sim_psf).astype(bool)))
    return (normalize_psf(sim_psf) - normalize_psf(meas_psf)).flatten()

def compute_strehl(image, model, cutout=100):
    
    # compute model ratio
    model_cutout = take_cutout(model, cutout=cutout)
    model_sum = np.sum(model_cutout)
    model_peak = np.max(model_cutout)
    model_normpeak = model_peak / model_sum
    
    # compute measured ratio
    meas_cutout = take_cutout(image, cutout=cutout)
    meas_sum = np.sum(meas_cutout)
    meas_peak = np.max(meas_cutout)
    meas_normpeak = meas_peak / meas_sum
    
    # Strehl = measured ratio / model ratio
    return meas_normpeak / model_normpeak

def take_cutout(image, cutout=100):
    y, x = np.where(image == image.max())
    lower = lambda x: x if x > 0 else 0
    return deepcopy(image)[lower(y[0]-cutout//2):y[0]+cutout//2, lower(x[0]-cutout//2):x[0]+cutout//2]


def get_hadamard_modes(dm_mask, roll=0, shuffle=None):
    nact = np.count_nonzero(dm_mask)
    if shuffle is None:
        shuffle = slice(len(nact))
    np2 = 2**int(np.ceil(np.log2(nact)))
    print(f'Generating a {np2}x{np2} Hadamard matrix.')
    hmat = hadamard(np2)
    return np.roll(hmat[shuffle,:nact], roll, axis=1)
    '''cmds = []
    nact = np.count_nonzero(dm_mask)
    for n in range(nact):
        cmd = np.zeros(nact)
        cmd[n] = 1
        cmds.append(cmd)
    return np.asarray(cmds)'''

def get_response_measurements(camstream, dmstream, modes, scale_factor, phase_diversity):
    measurements = []
    for m in modes:
        # send command to dm
        camstream.dm.state = m # replace me
        # optional sleep
        # measure on camera
        measurements.append(camstream.get_latest())
        
        camstream.dm.state = m + phase_diversity # replace me
        # optional sleep
        # measure on camera
        measurements.append(camstream.get_latest())
        
    return measurements

def measure_hadamard(camstream, dmstream, dm_mask, scale_factor, phase_diversity, do_pm=True):
    hmodes = get_hadamard_modes(dm_mask)
    if do_pm:
        hmodes_neg = -1 * hmodes
        hmodes = np.vstack([hmodes.copy(), hmodes_neg])
        
    measured = get_response_measurements(camstream, dmstream, hmodes, scale_factor, phase_diversity)
    return hmodes, np.asarray(measured)

def fast_and_furious(psf1, psf2, eta=1.0):
    pass

def take_hadamard_measurements(dm_pupil, camstream, scale_factor, value=0.5):

    # take the hadamard measurements
    zbasis = zernike.arbitrary_basis(camstream.dm.circmapping, outside=0., nterms=10)[1:] 
    coeffs = np.random.normal(scale=0.01, size=len(zbasis))
    diff_phase_est = np.sum([c*z for c,z in zip(coeffs,zbasis)],axis=0).T[camstream.dm.circmapping] # transpose because of DM mapping. NEED TO ACCO

    hmodes, hpsfs = measure_hadamard(camstream, None, camstream.dm.circmapping, scale_factor, diff_phase_est)
    
    return hmodes, hpsfs

def estimate_phases_from_hadamard_measurements(hpsfs, fit_pupil, scale_factor, eta=1.0):
    model_psf = simulate_psf(fit_pupil, 1., 0., 0, add_noise=False)

    camstream.dm.state = diff_phase_est
    phase_div = camstream.dm.get_surface(scale=scale_factor)

    hphases_pm = []
    for psf_had, psf_had_pd in zip(hpsfs[::2], hpsfs[1::2]):

        # estimate strehl
        strehl = compute_strehl(psf_had, model_psf)

        # estimate phase
        phase_est = ff_estimate_phase(psf_had, psf_had_pd, phase_div, pupil=fit_pupil, eta=eta, S=strehl)
        hphases_pm.append(phase_est)
    hphases_pm = np.asarray(hphases_pm)
    hphases = (hphases_pm[:len(hphases_pm) // 2] - hphases_pm[len(hphases_pm) // 2:]) / 2. # difference the plus/minus
    return hphases

def cut_and_pad_image(image, cutlen=256, padlen=128):
    return pad(take_cutout(image, cutlen), padlen)

def get_response_measurements(camstream, dmstream, modes, phase_diversity, dm_mask, nimages=1):
    measurements = []
    for i, m in enumerate(modes):
        #m = m.copy() - np.mean(m[dm_mask])
        # send command to dm
        dmstream.write(m.astype(dmstream.buffer.dtype))
        sleep(0.01)
        # measure on camera
        measurements.append(np.mean(camstream.grab_many(nimages),axis=0))
        
        dmstream.write((m+phase_diversity[i]).astype(dmstream.buffer.dtype))
        sleep(0.01)
        # measure on camera
        measurements.append(np.mean(camstream.grab_many(nimages),axis=0))
        
    dmstream.write(np.zeros(dmstream.buffer.shape, dtype=dmstream.buffer.dtype))
        
    return measurements

def measure_hadamard(camstream, dmstream, dm_mask, phase_diversity, phase_bias=None, hvalue=0.05, nimages=1, do_pm=True, roll=0, shuffle=None):
    hmodes = get_hadamard_modes(dm_mask, roll=roll, shuffle=shuffle) * hvalue
    if do_pm:
        hmodes_neg = -1 * hmodes
        hmodes = np.vstack([hmodes.copy(), hmodes_neg])
        
    hcmds = []
    for h in hmodes:
        hcmds.append(map_vector_to_square_ALPAO(h))
    hcmds = np.asarray(hcmds)
    
    if phase_bias is not None:
        hcmds = hcmds + phase_bias
        
    measured = get_response_measurements(camstream, dmstream, hcmds, phase_diversity, dm_mask, nimages=nimages)
    return hcmds, np.asarray(measured)

def vector_to_2d(vector, mask):
    arr = np.zeros(mask.shape)
    arr[mask] = vector
    return arr

def estimate_phases_from_hadamard_measurements(hpsfs, fit_pupil, phase_diversity, eta=1.0):
    wavelength = .650
    microns_surface_to_rad_wavefront = 4*np.pi / wavelength
    model_psf = normalize_psf(simulate_psf(fit_pupil, 1., 0., 0, add_noise=False))

    hphases_pm = []
    for i, (psf_had, psf_had_pd) in enumerate(zip(hpsfs[::2], hpsfs[1::2])):
        psf_had = cut_and_pad_image(psf_had).astype(float)
        psf_had -= np.median(psf_had)
        psf_had = normalize_psf(psf_had) * np.sum(fit_pupil)
        psf_had_pd = cut_and_pad_image(psf_had_pd).astype(float)
        psf_had_pd -= np.median(psf_had_pd)
        psf_had_pd = normalize_psf(psf_had_pd) * np.sum(fit_pupil)

        # estimate strehl
        strehl = compute_strehl(normalize_psf(psf_had), model_psf)

        # estimate phase
        phase_est = ff_estimate_phase(window_image(shift_to_centroid(psf_had)), window_image(shift_to_centroid(psf_had_pd)), phase_diversity[i],
                                      pupil=pupil, eta=eta, S=strehl)

        hphases_pm.append(phase_est)
    hphases_pm = np.asarray(hphases_pm)
    hphases = (hphases_pm[:len(hphases_pm) // 2] - hphases_pm[len(hphases_pm) // 2:]) / 2. # difference the plus/minus
    return hphases, hphases_pm

def estimate_phases_from_hadamard_measurements_nopd(hpsfs, fit_pupil, eta=1.0):
    wavelength = .650
    microns_surface_to_rad_wavefront = 4*np.pi / wavelength
    model_psf = normalize_psf(simulate_psf(fit_pupil, 1., 0., 0, add_noise=False))

    hphases_pm = []
    for psf_had, psf_had_pd in zip(hpsfs[::2], hpsfs[1::2]):
        psf_had = cut_and_pad_image(psf_had).astype(float)
        psf_had -= np.median(psf_had)
        psf_had = normalize_psf(psf_had) * np.sum(fit_pupil)
        # estimate strehl
        strehl = compute_strehl(normalize_psf(psf_had), model_psf)

        # estimate phase
        phase_est = ff_estimate_phase(shift_to_centroid(psf_had), None, None,
                                      pupil=pupil, eta=eta, S=strehl)

        hphases_pm.append(phase_est)
    hphases_pm = np.asarray(hphases_pm)
    hphases = (hphases_pm[:len(hphases_pm) // 2] - hphases_pm[len(hphases_pm) // 2:]) / 2. # difference the plus/minus
    return hphases, hphases_pm

def ff_estimate_phase(psf1, psf2, diff_phase, eta=10000, a=None, pupil=None, S=None,):
    
    if a is None:
        #a = get_a(pupil)
        a = get_a(pupil_sm)
    if S is None:
        S = 0.8 #estimate_strehl
        
    p1_even, p1_odd = get_even_odd_fft(psf1)
    if psf2 is not None:
        p2_even, p2_odd = get_even_odd_fft(psf2)

    y = get_y(p1_odd, a, eta)
    magv = get_magv(p1_even, S, a, y)
    
    if psf2 is not None:
        pd_even, pd_odd = get_even_odd_fft(diff_phase)
        yd = fft2_shiftnorm(pd_odd*pupil_sm)*1j 
        vd = fft2_shiftnorm(pd_even*pupil_sm)
        sgnv = get_sgnv(p2_even, p1_even, vd, yd, y)
    else:
        sgnv = np.sign(a)
        
    #return sgnv, magv, y, yd, vd, p1_even, p1_odd, p2_even, p2_odd, pd_even, pd_odd

    phase_est = get_phase_estimate(sgnv, magv, y).real
    phase_est -= np.mean(phase_est[pupil.astype(bool)])
    phase_est *= pupil.astype(bool)
    return phase_est

def get_if_mask(cenyx, radius, shape):
    mask = np.zeros(shape, dtype=bool)
    circ = draw.circle(cenyx[0][0], cenyx[1][0], radius, shape=shape)
    mask[circ] = 1
    return mask

def fit_zernikes(image, zbasis, mask):
    ngood = np.count_nonzero(mask)
    return [np.sum((image * b)[mask])/ngood for b in zbasis]

def surface_from_zcoeffs(coeffs, zbasis):
    return np.sum([c*z for c,z in zip(coeffs,zbasis)],axis=0)

def grab_nearest(slaved, dm_mask):
    nearest = []
    slaved_idx = np.where(slaved)[0]
    not_slaved = ~slaved
    for s in slaved_idx:
        s_vec = np.zeros(97)
        s_vec[s] = 1
        s_map = map_vector_to_square_ALPAO(s_vec).astype(bool)
        s_loc = np.where(s_map)
        distances = get_distance(s_loc, dm_mask)
        nearest_act = distances[not_slaved] == distances[not_slaved].min()
        nearest.append(np.where(not_slaved)[0][nearest_act][0])
    return nearest
        
def get_distance(locyx, mask):
    idy, idx = np.indices(mask.shape)
    idy -= locyx[0]
    idx -= locyx[1]
    distance = np.sqrt(idy**2 + idx**2)
    return map_square_to_vector_ALPAO(distance)
     
def fill_in_slaved_cmds(cmd_vec, slaved_vec_idx, neighbor_mapping):
    cmd = cmd_vec.copy()
    for slaved, neighbor in zip(slaved_vec_idx, neighbor_mapping):
        cmd[slaved] = cmd[neighbor]
    return cmd