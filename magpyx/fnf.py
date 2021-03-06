'''
To do:
* Make functions for closed loop / end-to-end operations
* Add option for using DM model instead of measured IFs
* Add option for modal approach instead of IFs?
* Add command-line hooks of the form:
>> fnf camsci1 dmncpc --if_cube "/opt/MagAOX/etc/mygreatIFs.fits" --filter 1.4
>> fnf_geometric_calib camsci1 dmncpc --nrepeats 5 --sdflkj
>> fnf_if_calib camsci1 dmncpc --nrepeats 5 --sdflkj

* Test by getting working on the camscis with the NCPC DM?
* optimize (replace FFTs in odd/even calc, make centroiding optional, etc)
* need to figure out structure for saving/loading calibration products
'''

from time import sleep
from copy import deepcopy

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.ndimage.interpolation import shift
from scipy.optimize import leastsq
from scipy.linalg import hadamard
from skimage import draw
import poppy
from poppy import zernike

from magpyx.utils import ImageStream
from magpyx.dm.t2w_offload import pseudoinverse_svd
#import dm_model as model

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('f&f')

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

def shift_to_centroid(im, shiftyx=None, order=3):
    if shiftyx is None:
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

def get_magaox_pupil(npix, rotation=38.75, grid_size=6.5, sm=False):
    pupil_diam = 6.5 #m
    secondary = 0.293 * pupil_diam
    primary = poppy.CircularAperture(radius=pupil_diam/2.)
    sec = poppy.AsymmetricSecondaryObscuration(secondary_radius=secondary/2.,
                                                 #support_angle=(45, 135, 225, 315),
                                                 #support_width=[0.01905,]*4,
                                                 #support_offset_y=[0, -0.34, 0.34, 0],
                                                 rotation=rotation,
                                                 name='Complex secondary')
    opticslist = [primary,]
    if sm:
        opticslist.append(sec)
    pupil = poppy.CompoundAnalyticOptic( opticslist=opticslist, name='Magellan')
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

def fit_pupil_to_psf(camstream, nimages, padding=0, fwhm_guess=10):
    # measure an image
    image = np.mean(camstream.grab_many(nimages),axis=0).astype(float)
    meas_psf = image - np.median(image)
    meas_psf = shift_to_centroid(meas_psf)
    meas_psf = pad(meas_psf, padding)
    scale_factor, shifty, shiftx = fit_psf(meas_psf, get_magaox_pupil, fwhm_guess=fwhm_guess)[0]
    pupil = get_magaox_pupil(meas_psf.shape[0], grid_size=6.5*scale_factor)
    return pupil, scale_factor, shifty, shiftx  

def fit_psf(meas_psf, pupil_func, fwhm_guess=10.):
    shape = meas_psf.shape[0]
    pupil_guess = fwhm_guess / 2. 
    return leastsq(psf_err, [pupil_guess, 0, 0], args=(pupil_func, shape, meas_psf), epsfcn=0.01)

def psf_err(params, pupil_func, sampling , meas_psf):
    scale_factor, ceny, cenx = params
    pupil = pupil_func(sampling, grid_size=6.5*scale_factor)
    
    sim_psf = shift_to_centroid(simulate_psf(pupil, 1, 0, np.zeros((sampling, sampling)), add_noise=False), (ceny, cenx))

    #print(rms(sim_psf-meas_psf, np.ones_like(sim_psf).astype(bool)))
    return (normalize_psf(sim_psf) - normalize_psf(meas_psf)).flatten()

def fit_dm_rotation_ellipticity(camstream, dmstream, dm_mask, nimages, cmd_value=0.5):
    
    # get tip/tilt terms
    zbasis_dm = zernike.arbitrary_basis(dm_mask, outside=0., nterms=3)[1:] 
    
    # measure psf displacement from tip/tilt
    value = cmd_value

    # measure
    tt_ims = []
    tt_ims.append(np.mean(camstream.grab_many(nimages), axis=0)) #ref
    sleep(1.0)
    for tt in zbasis_dm:
        dmstream.write((tt * value).astype(dmstream.buffer.dtype))
        sleep(1.0)
        tt_ims.append(np.mean(camstream.grab_many(nimages), axis=0))
    dmstream.write(np.zeros(dmstream.buffer.shape, dtype=dmstream.buffer.dtype))

    # compute displacements
    tt_coms = []
    for im in tt_ims:
        im_com = np.squeeze(np.where(im == im.max()))
        tt_coms.append(im_com)
    tt_coms = np.vstack(tt_coms)
    tt_coms -= tt_coms[0]

    # get angles of displacement
    angle1 = np.arctan2(tt_coms[1][0], tt_coms[1][1])
    angle2 = np.arctan2(tt_coms[2][0], tt_coms[2][1]) - np.pi/2.
    mean_angle = (angle1 + angle2) / 2.
    logger.info(f'Found rotation: {mean_angle} rad, {np.rad2deg(mean_angle)} deg')

    # get ratio of displacement
    disp1 = np.sqrt(tt_coms[1][0]**2 + tt_coms[1][1]**2)
    disp2 = np.sqrt(tt_coms[2][0]**2 + tt_coms[2][1]**2)
    disp_ratio = disp1 / disp2
    logger.info(f'Found x/y displacement ratio: {disp_ratio}')
    
    return mean_angle, disp_ratio

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

def take_cutout(image, cutout=100, cenyx=None):
    if cutout is None:
        return image
    if cenyx is None:
        y = int(np.rint((image.shape[0] - 1) / 2.))
        x = int(np.rint((image.shape[0] - 1) / 2.))
    else:
        y, x = cenyx
    lower = lambda x: x if x > 0 else 0
    return deepcopy(image)[lower(y-cutout//2):y+cutout//2, lower(x-cutout//2):x+cutout//2]

def cut_and_pad_image(image, cutlen=256, padlen=128):
    return pad(take_cutout(image, cutlen), padlen)

def get_response_measurements(camstream, dmstream, modes, phase_diversity, dm_mask, nimages=1, padding=0):
    measurements = []
    for i, m in enumerate(modes):
        #m = m.copy() - np.mean(m[dm_mask])
        # send command to dm
        dmstream.write(m.astype(dmstream.buffer.dtype))
        sleep(0.01)
        # measure on camera
        im = np.mean(camstream.grab_many(nimages),axis=0).astype(float)
        im -= np.median(im)
        im = pad(im, padding)
        measurements.append(im)
        
        dmstream.write((m+phase_diversity[i]).astype(dmstream.buffer.dtype))
        sleep(0.01)
        # measure on camera
        im = np.mean(camstream.grab_many(nimages),axis=0).astype(float)
        im -= np.median(im)
        im = pad(im, padding)
        measurements.append(im)
        
    dmstream.write(np.zeros(dmstream.buffer.shape, dtype=dmstream.buffer.dtype))
        
    return measurements

def measure_hadamard(camstream, dmstream, dm_map, dm_mask, phase_diversity, padding=0, phase_bias=None, hvalue=0.05, nimages=1, do_pm=True, roll=0, shuffle=None):
    hmodes = get_hadamard_modes(dm_mask, roll=roll, shuffle=shuffle) * hvalue
    if do_pm:
        hmodes_neg = -1 * hmodes
        hmodes = np.vstack([hmodes.copy(), hmodes_neg])
        
    hcmds = []
    for h in hmodes:
        hcmds.append(map_vector_to_square(h, dm_map, dm_mask))
    hcmds = np.asarray(hcmds)
    
    if phase_bias is not None:
        hcmds = hcmds + phase_bias
        
    measured = get_response_measurements(camstream, dmstream, hcmds, phase_diversity, dm_mask, nimages=nimages, padding=padding)
    return hcmds, np.asarray(measured)

def vector_to_2d(vector, mask):
    arr = np.zeros(mask.shape)
    arr[mask] = vector
    return arr

def estimate_phases_from_hadamard_measurements(hpsfs, fit_pupil, phase_diversity, microns_surface_to_rad_wavefront, padding=0, eta=1.0):
    model_psf = normalize_psf(simulate_psf(fit_pupil, 1., 0., 0, add_noise=False))
    hphases_pm = []
    for i, (psf_had, psf_had_pd) in enumerate(zip(hpsfs[::2], hpsfs[1::2])):
        psf_had -= np.median(psf_had)
        psf_had = pad(psf_had, padding)
        psf_had = normalize_psf(psf_had) * np.sum(fit_pupil)
        psf_had_pd -= np.median(psf_had_pd)
        psf_had_pd = pad(psf_had_pd, padding)
        psf_had_pd = normalize_psf(psf_had_pd) * np.sum(fit_pupil)

        # estimate strehl
        strehl = compute_strehl(normalize_psf(psf_had), model_psf)

        # estimate phase
        phase_est = ff_estimate_phase(shift_to_centroid(psf_had), shift_to_centroid(psf_had_pd), phase_diversity[i],
                                      pupil=fit_pupil, eta=eta, S=strehl)

        hphases_pm.append(phase_est)
    hphases_pm = np.asarray(hphases_pm)
    hphases = (hphases_pm[:len(hphases_pm) // 2] - hphases_pm[len(hphases_pm) // 2:]) / 2. # difference the plus/minus
    return hphases, hphases_pm

def measure_interaction_matrix(camstream, dmstream, dm_map, dm_mask, zbasis_dm, zbasis_pupil, microns_surface_to_rad_wavefront, fit_pupil, pd_cmd, pd_meas, nimages=50, padding=0, nrepeats=10, hval=0.025, coeff_scale=0.005, nmodes=5, eta=1e-3):
    
    hmodes = get_hadamard_modes(dm_mask)
    nhad = len(hmodes)
    
    # measure hadamard modes with phase diversity
    allhphases = []
    for n in range(nrepeats):
        logger.info(f'On Hadamard measurement sequence {n+1}/{nrepeats}!')

        # unique phase diversity for each measurement (including +/-)
        #allcoeffs = []
        #all_diff_cmds = []
        #all_diff_ests = []
        #for n in range(2*nhad):
        #    coeffs = np.random.normal(scale=coeff_scale, size=nmodes)
        #    diff_phase_cmd = -np.sum([c*z for c,z in zip(coeffs,zbasis_dm[2:nmodes+2])],axis=0)
        #    diff_phase_est = np.sum([c*z for c,z in zip(coeffs,zbasis_pupil[2:nmodes+2])],axis=0) * microns_surface_to_rad_wavefront
        #    all_diff_cmds.append(diff_phase_cmd)
        #    all_diff_ests.append(diff_phase_est)

        # replacing estimated phase diversity with fixed measured (via GS)
        all_diff_cmds = [pd_cmd,] * 2*nhad
        all_diff_ests = [pd_meas,] * 2*nhad

        # take the actual measurements (hadamard + phase diversity)
        hcmds, hpsfs = measure_hadamard(camstream, dmstream, dm_map, dm_mask, all_diff_cmds, padding=0, phase_bias=None,
                                         nimages=nimages, hvalue=hval, do_pm=True)
        return hpsfs, all_diff_ests
        # produce the estimates
        hphases, hphases_pm = estimate_phases_from_hadamard_measurements(hpsfs, fit_pupil, all_diff_ests, microns_surface_to_rad_wavefront, padding=padding, eta=eta)
        allhphases.append(hphases)
        
    # get IF matrix
    hphases_mean = np.mean(allhphases, axis=0) / hval
    #return hphases_mean
    hinv = np.linalg.inv(hmodes)
    ifmat = np.dot( hinv, hphases_mean.reshape(nhad,-1) )
    
    return ifmat, hphases_mean, hmodes

def get_cmat(ifmat, n_threshold=50):
    cmat, threshold, U, s, Vh = pseudoinverse_svd(ifmat, n_threshold=n_threshold)
    return cmat

def clean_ifmat(ifmat, zbasis_pupil, pupil_sm, imshape, ifrad=9):
    ifs_clean = []
    for curif in ifmat:
        curif2d = curif.reshape(imshape)
        cenyx = np.where(curif2d == curif2d.max())
        if_mask = get_if_mask(cenyx, 9, imshape)
        fit_z = surface_from_zcoeffs(fit_zernikes(curif2d*~if_mask, zbasis_pupil[:30], pupil_sm.astype(bool)), zbasis_pupil)
        ifs_clean.append((pupil_sm*(curif2d-fit_z)).flatten())
    ifs_clean = np.asarray(ifs_clean)
    return ifs_clean
    
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

def fnf_closed_loop(camstream, dmstream, ifmat, cmat, slavedmap, pupil, dm_mask, dm_map, model_psf, cenwave=0.8, gain=0.1, leak_gain=0.02, nimages=30, niter=None, centroid=True, eta=1e-3, delay=0., gauss_sigma=1.2):

    dm_shape = dm_mask.shape
    nact = dm_map.max()
    imshape = pupil.shape
    
    # initial phase estimate before entering the loop
    init_psf  = np.mean(camstream.grab_many(nimages).astype(float), axis=0) #- dark_median
    init_psf = cut_and_pad_image(init_psf)
    init_psf -= np.median(init_psf)
    init_psf = normalize_psf(init_psf) * np.sum(pupil)
    strehl = compute_strehl(normalize_psf(init_psf), model_psf)

    phase_est = gaussian_filter(ff_estimate_phase(init_psf, None, None, pupil=pupil, eta=eta, S=strehl), gauss_sigma)
    psf_curr = init_psf
    last_corr = 0
    cmd = np.zeros(dm_shape)

    #rms_vals = [rms(phase_est, pupil.astype(bool))]
    #strehls = [strehl,]
    #phase_ests = []
    
    try:
        if camstream.semindex is None:
            camstream.semindex = dmstream.getsemwaitindex(1)
            logger.info(f'Got semaphore index {camstream.semindex}.')
        # flush semaphores before entering loop
        camstream.semflush(camstream.semindex)
        while True:
            # wait for a semaphore, then start the iteration
            camstream.semwait(camstream.semindex)
            
            # -----all the loop stuff goes here-----
            
            # get actuator commands to correct residual phase
            delta_cmd_good = -np.dot(phase_est.flatten(), cmat)
            delta_cmd = np.zeros(nact)
            delta_cmd[good_vec] = delta_cmd_good
            delta_cmd = fill_in_slaved_cmds(delta_cmd, slaved_vec_idx, nearby)
            delta_cmd -= np.mean(delta_cmd)
            delta_cmd *= gain
            
            # leaky integrator
            cmd -= leak_gain*cmd
            cmd += map_vector_to_square(delta_cmd, dm_map, dm_mask)

            # apply the command
            dmstream.write((cmd).astype(dmstream.buffer.dtype))
            sleep(delay)

            # forward model to get DM response and to get phase div. input to FF
            delta_proj = np.dot(delta_cmd, ifs_clean).reshape(imshape)

            rms_val = rms(phase_est, pupil.astype(bool))
            #rms_vals.append(rms_val)

            # measure next psf
            psf_last = psf_curr
            psf_curr = np.mean(camstream.grab_many(nimages), axis=0)

            # process the psf
            psf_curr = cut_and_pad_image(psf_curr).astype(float)
            psf_curr -= np.median(psf_curr)
            psf_curr = normalize_psf(psf_curr) * np.sum(pupil)

            # estimate strehl
            strehl = compute_strehl(normalize_psf(psf_curr), model_psf)
            #strehls.append(strehl)

            # estimate phase
            phase_est = gaussian_filter(ff_estimate_phase(shift_to_centroid(psf_last), shift_to_centroid(psf_curr), delta_proj, pupil=pupil, eta=eta, S=strehl), gauss_sigma)
            #phase_ests.append(phase_est)
            
    except KeyboardInterrupt:
        logger.info('Caught a keyboard interrupt. Exiting F&F loop.')
