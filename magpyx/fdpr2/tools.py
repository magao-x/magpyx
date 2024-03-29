import configparser
from os import path, symlink, remove, mkdir
import pathlib
from datetime import datetime
from glob import glob

import numpy as np
from poppy import zernike

from astropy.io import fits
from skimage.filters.thresholding import threshold_otsu


from ..utils import ImageStream, create_shmim, str2bool
from ..instrument import take_dark
from ..dm.dmutils import get_hadamard_modes, map_vector_to_square
from ..dm import control
from ..imutils import rms, write_to_fits, remove_plane, center_of_mass, shift, remove_plane

from .estimation import multiprocess_phase_retrieval, run_phase_retrieval
from .measurement import get_probed_measurements, get_ref_measurement, get_response_measurements

from importlib import import_module

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr2')


try:
    import cupy as cp
except ImportError:
    logger.warning('Could not import cupy. You may lose functionality.')
    cp = None


SUBDIRS = ['ctrlmat', 'dmmap', 'dmmask', 'dmmodes', 'estrespM', 'ifmat',
           'measrespM', 'singvals', 'wfsmap', 'wfsmask', 'wfsmodes']

# ------ GENERAL -------

def phase_rms_to_cmd_rms(phase, wavelen):
    cmd_um = phase * wavelen / (4*np.pi) * 1e6
    return cmd_um

def cmd_rms_to_phase_rms(cmd, wavelen):
    phase = cmd * (4*np.pi) / wavelen * 1e-6 
    return phase

def get_fitting_region(shape, nside):
    cen = ( (shape[0]-1)/2., (shape[1]-1)/2.)
    yxslice = (slice( int(np.rint(cen[0]-nside/2.)), int(np.rint(cen[0]+nside/2.))),
               slice( int(np.rint(cen[1]-nside/2.)), int(np.rint(cen[1]+nside/2.))))
    mask = np.zeros(shape, dtype=bool)
    mask[yxslice] = 1
    return mask, yxslice #??
    
def get_defocus_probes(fitmask, probevals, wavelen, scalefactor=1):
    if cp is not None:
        fitmask = cp.array(fitmask)
        zmodes = cp.asnumpy(zernike.arbitrary_basis(fitmask, nterms=4, outside=0))
    else:
        zmodes = zernike.arbitrary_basis(fitmask, nterms=4, outside=0)

    #zmodes = zernike.arbitrary_basis(fitmask, nterms=4, outside=0)
    phasevals = cmd_rms_to_phase_rms(probevals, wavelen) * scalefactor
    return np.exp(1j*zmodes[-1]*phasevals[:,None,None])

def get_defocus_probe_cmds(dm_mask, probevals, config_params):
    if cp is not None:
        dm_mask = cp.array(dm_mask)
        zmodes = cp.asnumpy(zernike.arbitrary_basis(dm_mask, nterms=4, outside=0))
    else:
        zmodes = zernike.arbitrary_basis(dm_mask, nterms=4, outside=0)
    return probevals[:,None,None] * zmodes[-1]

def get_defocus_probe_cmds_magaox(dm_mask, probevals, config_params):
    # read in zbasis (need config path)
    zpath = config_params.get_param('estimation', 'div_path', str)
    with fits.open(zpath) as f:
        zmodes = f[0].data
    # pull out defocus mode and scale
    return probevals[:,None,None] * zmodes[2]

def get_centroid(camstream, navg=1, dmdelay=2, dark=None):
    imref = get_ref_measurement(camstream, navg, dmdelay)
    if dark is not None:
        imref -= dark
    refmask = imref > imref.max() * 0.5
    com_yx = center_of_mass(imref * refmask) # reference
    return com_yx

def translate_to_centroid(im, com_yx):
    shape = im.shape
    cen = ( (shape[0] - 1) /2., (shape[1] - 1) /2.)
    newim = shift(im, (cen[0]-com_yx[0], cen[1]-com_yx[1]))
    return newim

def translate_cube_to_centroid(imcube, com_yx):
    out = []
    for im in imcube:
        out.append(translate_to_centroid(im, com_yx))
    return np.asarray(out)

def translate_hypercube_to_centroid(hcube, com_yx):
    out = np.zeros_like(hcube)
    for i, imcube in enumerate(hcube):
        for j, im in enumerate(imcube):
            out[i,j] = translate_to_centroid(im, com_yx)
    return out

def get_amplitude_mask(amplitude, threshold_factor):
    thresh = threshold_otsu(amplitude)
    mask = amplitude > (thresh*threshold_factor)
    return mask
    
def get_strehl(phase, amplitude, mask):
    
    Efield = amplitude * np.exp(1j*phase)
    Efield /= np.sqrt(np.sum(Efield * Efield.conj()))

    amp = np.abs(Efield)
    phase = np.angle(Efield)
    log_amp = np.log(amp)
    varlogamp = np.var(log_amp[mask])

    varphase = np.var(phase[mask])  
    strehl_phase = np.exp(-varphase)
    strehl_amp = np.exp(-varlogamp)  
    return strehl_phase * strehl_amp, strehl_phase, strehl_amp

def test_defocus(dmstream, cmds):
    
    curcmd = dmstream.grab_latest()
    for cmd in cmds:
        dmstream.write(cmd)
        input("Press Enter to continue...")
    dmstream.write(curcmd)

# ----- CLOSED LOOP ------

def measure_and_estimate_phase_vector(camstream=None, dmstream=None, probe_cmds=None, fitmask=None, fitslice=None, wfsmask=None, Eprobes=None, tol=1e-7, reg=0, wreg=1e2, navg=1, dmdelay=2, dark=None, config_params=None, offset=0):


    if dark is None:
        dark = 0
    # get centroid from potentially saturated PSF (no defocus)
    com_yx = get_centroid(camstream, navg=navg, dmdelay=dmdelay, dark=dark)

    # measure defocused PSFS
    psfs = get_probed_measurements(camstream, dmstream, probe_cmds, navg=navg, dmdelay=dmdelay)
    if dark is not None:
        psfs -= dark

    # translate to calculated centroid
    psfs_cen = translate_cube_to_centroid(psfs, com_yx) - np.median(psfs)

    # run estimator
    estdict = run_phase_retrieval(psfs_cen, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True)

    phase = estdict['phase_est'] - offset
    amp = estdict['amp_est']
    # and remove tip/tilt
    #phase_pttrem = remove_plane(phase, pupilmask)

    # stack and apply wfsmask
    stacked = phase[fitslice]#np.concatenate([phase[fitslice], amp[fitslice]], axis=1)
    stacked = remove_plane(stacked, wfsmask)

    # update shmims
    # threshold phase based on amplitude? (reject phase values where amplitude is < some threshold)
    threshold_factor = config_params.get_param('control', 'ampthreshold', float)
    amp_mask = get_amplitude_mask(amp, threshold_factor)
    amp_norm = amp / np.mean(amp[amp_mask])
    phase0 = remove_plane(phase, amp_mask) * amp_mask
    update_estimate_shmims(phase0, amp, config_params)

    # --- estimate Strehl ratio ---
    phase_rms = np.std(phase0[amp_mask])#rms(phase, pupil)
    amp_rms = np.std(amp_norm[amp_mask])#rms(amp_norm, pupil)
    amp_lnrms = np.std(np.log(amp_norm)[amp_mask])#rms(np.log(amp_norm), pupil)

    strehl, strehl_phase, strehl_amp = get_strehl(phase0, amp_norm, amp_mask)
    #strehl = np.exp(-phase_rms**2) * np.exp(-amp_lnrms**2)

    logger.info(f'Estimated phase RMS: {phase_rms:.3} (rad)')
    logger.info(f'Estimated amplitude RMS: {amp_rms*100:.3} (%)')
    logger.info(f'Estimated Strehl: {strehl:.2f} ({strehl_phase:.2f} phase-only and {strehl_amp:.2f} amplitude-only)')

    return stacked[wfsmask]

def close_loop(config_params):
    
    # open shmims
    dmctrlstream = ImageStream(config_params.get_param('control', 'dmctrlchannel', str))
    dmdivstream = ImageStream(config_params.get_param('diversity', 'dmdivchannel', str))
    camname = config_params.get_param('camera', 'name', str)
    camstream = ImageStream(camname)
    dark_name = config_params.get_param('diversity', 'dark_shmim', str)
    if dark_name.lower() != 'none':
        darkstream = ImageStream(dark_name)
        dark = darkstream.grab_latest()
    else:
        dark = 0

    # get measurement and estimation parameters
    navg = config_params.get_param('diversity', 'navg', int)
    dmdelay = config_params.get_param('diversity', 'dmdelay', int)
    probevals = np.asarray(config_params.get_param('diversity', 'probevals', float))
    wavelen = config_params.get_param('estimation', 'wavelen', float)
    scalefactor = config_params.get_param('estimation', 'scalefactor', float)
    N = config_params.get_param('estimation', 'N', int)
    nside = config_params.get_param('estimation', 'Nfit', int)
    tol = config_params.get_param('estimation', 'tol0', float)
    reg = config_params.get_param('estimation', 'reg', float)
    wreg = config_params.get_param('estimation', 'wreg', float)
    

    fitmask, fitslice = get_fitting_region((N,N), nside)
    Eprobes = get_defocus_probes(fitmask, probevals, wavelen, scalefactor=scalefactor)

    # dm actuator mapping
    with fits.open(config_params.get_param('interaction', 'dm_map', str)) as f:
        dm_map = f[0].data
    with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
        dm_mask = f[0].data

    # get diversity probes func
    calib_func = config_params.get_param('estimation', 'div_func', str)
    mname, fname = calib_func.rsplit('.', 1)
    mod = import_module(mname)
    probe_func = getattr(mod, fname)
    probe_cmds = probe_func(dm_mask, probevals, config_params)

    # get other relevant parameters for closed loop
    niter = config_params.get_param('control', 'niter', int)
    gain = config_params.get_param('control', 'gain', float)
    leak = config_params.get_param('control', 'leak', float)
    delay = config_params.get_param('control', 'delay', float)

    # get the ctrl matrix
    calibpath = config_params.get_param('calibration', 'path', str)
    with fits.open(path.join(calibpath, 'ctrlmat.fits')) as f:
        ctrlmat = f[0].data

    # get the wfs mask
    with fits.open(path.join(calibpath, 'wfsmask.fits')) as f:
        wfsmask = f[0].data.astype(bool)

    # dm actuator mapping
    with fits.open(config_params.get_param('interaction', 'dm_map', str)) as f:
        dm_map = f[0].data
    with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
        dm_mask = f[0].data

    # set up the wfs function
    wfsfuncdict = {
        'camstream' : camstream,
        'dmstream' : dmdivstream,
        'probe_cmds' : probe_cmds,
        'fitmask' : fitmask,
        'fitslice' : fitslice,
        'wfsmask' : wfsmask,
        'Eprobes' : Eprobes,
        'tol' : tol,
        'reg' : reg,
        'wreg' : wreg,
        'navg' : navg,
        'dmdelay' : dmdelay,
        'dark' : dark,
        'config_params' : config_params
    }

    wfsfunc = measure_and_estimate_phase_vector
    
    control.closed_loop(dmctrlstream, ctrlmat, wfsfunc, dm_map, dm_mask, niter=niter, gain=gain,
                        leak=leak, delay=delay, paramdict=wfsfuncdict)

    # close all the things
    dmdivstream.close()
    dmctrlstream.close()
    camstream.close()


# ----- ESTIMATION HELPERS ------

def compute_control_matrix(config_params, nmodes=None, write=True):
    '''
    Take the Hadamard response matrix and compute:
    * IF matrix
    * wfsmap, wfsmask
    * dmmap, dmmask
    * SVD of IF matrix to get dmmodes, singvals, wfsmodes
    * interpolation of dmmodes based on dmmask
    * control matrix
    '''

    # get hmeas, hmodes, and hval
    calib_path = config_params.get_param('calibration', 'path', str)
    hmeas_path = path.join(calib_path, 'estrespM.fits')
    with fits.open(hmeas_path) as f:
        hmeas = f[0].data
    nact = config_params.get_param('interaction', 'nact', int)
    hmodes = get_hadamard_modes(nact)
    hval = config_params.get_param('interaction', 'hval', float)

    # get dm map and mask and thresholds
    with fits.open(config_params.get_param('interaction', 'dm_map', str)) as f:
        dm_map = f[0].data
    with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
        dm_mask = f[0].data.astype(bool)
    dmthresh = config_params.get_param('control', 'dmthreshold', float)
    wfsthresh = config_params.get_param('control', 'wfsthreshold', float)
    ninterp = config_params.get_param('control', 'ninterp', int)
    #npix = config_params.get_param('control', 'npix', int)

    # reduce measured data to npix region
    N = config_params.get_param('estimation', 'N', int)
    nside = config_params.get_param('estimation', 'Nfit', int)
    #fitmask, fitslice = get_fitting_region((N,N), nside)
    #slicezyx = (slice(None),*fitslice) # skip reference measurement
    #hmeas = hmeas[slicezyx]

    if nmodes is None:
        nmodes = config_params.get_param('control', 'nmodes', int)

    remove_modes = config_params.get_param('control', 'remove_modes', int)
    if remove_modes == 0:
        remove_modes = None
    tikreg = config_params.get_param('control', 'tikreg', float)
    regtype = config_params.get_param('control', 'regtype', str)

    ctrldict = control.get_control_matrix_from_hadamard_measurements(hmeas,
                                                                     hmodes,
                                                                     hval,
                                                                     dm_map,
                                                                     dm_mask,
                                                                     wfsthresh=wfsthresh,
                                                                     dmthresh=dmthresh,
                                                                     ninterp=ninterp,
                                                                     nmodes=nmodes,
                                                                     remove_modes=remove_modes,
                                                                     regtype=regtype,
                                                                     treg=tikreg)

    date = datetime.now().strftime("%Y%m%d-%H%M%S")

    # write these out to file (should this be moved to the function in the console module?)
    if write:
        for key, value in ctrldict.items():
            outdir = path.join(calib_path, key)
            if not path.exists(outdir):
                mkdir(outdir)
            outname = path.join(calib_path, key, f'{key}_{date}.fits')
            write_to_fits(outname, value)
            logger.info(f'Wrote out {outname}')
            # update symlinks
            sympath = path.join(calib_path, key+'.fits')
            replace_symlink(sympath, outname)

    return ctrldict

def measure_response_matrix(config_params):

    # config params
    navg = config_params.get_param('diversity', 'navg', int)
    navg_ref = config_params.get_param('diversity', 'navg_ref', int)
    probevals = np.asarray(config_params.get_param('diversity', 'probevals', float))
    dmdelay = config_params.get_param('diversity', 'dmdelay', int)

    dmdivstream = ImageStream(config_params.get_param('diversity', 'dmdivchannel', str))
    dmstream = ImageStream(config_params.get_param('diversity', 'dmchannel', str))
    camstream = ImageStream(config_params.get_param('camera', 'name', str))
    dark_name = config_params.get_param('diversity', 'dark_shmim', str)
    if dark_name != 'None':
        darkstream = ImageStream(dark_name)
        dark = darkstream.grab_latest()
    else:
        dark = 0

    with fits.open(config_params.get_param('interaction', 'dm_map', str)) as f:
        dm_map = f[0].data
    with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
        dm_mask = f[0].data

    #probe_cmds = get_defocus_probe_cmds(dm_mask, probevals)
    # get diversity probes func
    calib_func = config_params.get_param('estimation', 'div_func', str)
    mname, fname = calib_func.rsplit('.', 1)
    mod = import_module(mname)
    probe_func = getattr(mod, fname)
    probe_cmds = probe_func(dm_mask, probevals, config_params)

    # take reference measurements first (1. in-focus image for centroiding, 2. defocus but no DM command)
    camstream = ImageStream(config_params.get_param('camera', 'name', str))
    logger.info('Taking centroid measurement')
    com_yx = get_centroid(camstream, navg=navg_ref, dmdelay=dmdelay)

    # take defocused images w/ no modal probe
    logger.info('Taking diversity-only (no modal commands) measurements for estimation')
    Imeasref = get_probed_measurements(camstream, dmdivstream, probe_cmds, navg=navg_ref, dmdelay=dmdelay)
    Imeasref -= dark

    # get the hadamard modes
    nact = config_params.get_param('interaction', 'nact', int)
    hval = config_params.get_param('interaction', 'hval', float)
    hmodes = get_hadamard_modes(nact)

    # reshape for DM
    hmodes_sq = np.asarray([map_vector_to_square(cmd, dm_map, dm_mask) for cmd in hmodes])
    # +/- and scaling
    dm_cmds = np.concatenate([hmodes_sq, -hmodes_sq]) * hval

    # get diversity cmds

    logger.info(f'Got a {hmodes.shape} Hadmard matrix and constructed a {dm_cmds.shape} DM command sequence.')
    logger.info(f'Taking measurements for interaction matrix...')
    imcube = get_response_measurements(camstream, dmstream, dmdivstream, probe_cmds, dm_cmds, navg=navg, dmdelay=dmdelay)
    imcube -= dark

    # centroid and stack all measurements
    Imeasref_cen = translate_cube_to_centroid(Imeasref, com_yx) - np.median(Imeasref)
    imcube_cen = translate_hypercube_to_centroid(imcube, com_yx) - np.median(imcube, axis=(-2,-1))[:,:,None,None]

    # stack reference before modal measurements
    outcube = np.concatenate([Imeasref_cen[None,:,:,:], imcube_cen], axis=0)

    camstream.close()

    return outcube # Nmodes + 1 x Ndiv x Nypix x Nxpix

def estimate_response_matrix(Iref, Icube, fitmask, tol0, tol1, reg, wreg, probevals, wavelen, scalefactor=1, processes=2, gpus=None):

    # get Eprobe (I don't like this here --- ugh)
    Eprobes = get_defocus_probes(fitmask, probevals, wavelen, scalefactor)
    
    # process first element of cube (no DM cmd) 
    logger.info('Running PR on reference measurement')
    outdict0 = run_phase_retrieval(Iref, fitmask, tol0, reg, wreg, Eprobes, init_params=None, bounds=True)

    # then process the hadamard cube
    logger.info('Running PR on Hadamard cube')
    init_params = outdict0['fit_params']
    rlist = multiprocess_phase_retrieval(Icube, fitmask, tol1, reg, wreg, Eprobes, init_params=init_params, bounds=False, processes=processes, gpus=gpus)
    # turn list of dictionaries into dictionary of lists
    return outdict0, {k: [cdict[k] for cdict in rlist] for k in rlist[0]}

def estimate_oneshot(config_params, update_shmim=True, write_out=False, skip_estimation=False, camstream_in=None, dmdivstream_in=None, dm_map=None, dm_mask=None):

    # open shmims
    #dmstream = ImageStream(config_params.get_param('diversity', 'dmchannel', str))
    camname = config_params.get_param('camera', 'name', str)
    if camstream_in is None:
        camstream = ImageStream(camname)
    else:
        camstream = camstream_in
    dark_name = config_params.get_param('diversity', 'dark_shmim', str)
    if dark_name != 'None':
        darkstream = ImageStream(dark_name)
        dark = darkstream.grab_latest()
    else:
        dark = 0

    # get measurement and estimation parameters
    if dmdivstream_in is None:
        dmdivstream = ImageStream(config_params.get_param('diversity', 'dmdivchannel', str))
    else:
        dmdivstream = dmdivstream_in
    navg = config_params.get_param('diversity', 'navg_ref', int)
    dmdelay = config_params.get_param('diversity', 'dmdelay', int)
    probevals = np.asarray(config_params.get_param('diversity', 'probevals', float))
    wavelen = config_params.get_param('estimation', 'wavelen', float)
    scalefactor = config_params.get_param('estimation', 'scalefactor', float)
    N = config_params.get_param('estimation', 'N', int)
    nside = config_params.get_param('estimation', 'Nfit', int)
    tol = config_params.get_param('estimation', 'tol0', float)
    reg = config_params.get_param('estimation', 'reg', float)
    wreg = config_params.get_param('estimation', 'wreg', float)
    modes = config_params.get_param('estimation', 'modes', str)
    if modes == 'None':
        modes = None

    fitmask, fitslice = get_fitting_region((N,N), nside)
    Eprobes = get_defocus_probes(fitmask, probevals, wavelen, scalefactor=scalefactor)

    # dm actuator mapping
    if dm_map is None:
        with fits.open(config_params.get_param('interaction', 'dm_map', str)) as f:
            dm_map = f[0].data
    if dm_mask is None:
        with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
            dm_mask = f[0].data

    #probe_cmds = get_defocus_probe_cmds(dm_mask, probevals)
    # get diversity probes func
    calib_func = config_params.get_param('estimation', 'div_func', str)
    mname, fname = calib_func.rsplit('.', 1)
    mod = import_module(mname)
    probe_func = getattr(mod, fname)
    probe_cmds = probe_func(dm_mask, probevals, config_params)

    # take reference image (no defocus) for centroid
    com_yx = get_centroid(camstream, navg=navg, dmdelay=dmdelay, dark=dark)

    # take defocused images
    Imeas = get_probed_measurements(camstream, dmdivstream, probe_cmds, navg=navg, dmdelay=dmdelay)
    Imeas -= dark

    # shift to centroid
    Imeas_cen = translate_cube_to_centroid(Imeas, com_yx) - np.median(Imeas)

    #return Imeas_cen, fitmask, tol, reg, wreg, Eprobes

    # run estimation
    if not skip_estimation:
        fitdict = run_phase_retrieval(Imeas_cen, fitmask, tol, reg, wreg, Eprobes, init_params=None, bounds=True, modes=modes)

        #return fitdict

        # report (maybe tell the user RMS, Strehl, etc. and tell them what shmims/files are updated)
        amp = fitdict['amp_est']

        # threshold phase based on amplitude? (reject phase values where amplitude is < some threshold)
        threshold_factor = config_params.get_param('control', 'ampthreshold', float)
        amp_mask = get_amplitude_mask(amp, threshold_factor)
        amp_norm = amp / np.mean(amp[amp_mask])

        #if shmim_mask is not None:
        #    pupil = pupil * shmim_mask # what is this???

        fitdict['phase_est'] *= amp_mask
        phase = remove_plane(fitdict['phase_est'], amp_mask) * amp_mask

        phase_rms = np.std(phase[amp_mask])#rms(phase, pupil)
        amp_rms = np.std(amp_norm[amp_mask])#rms(amp_norm, pupil)
        amp_lnrms = np.std(np.log(amp_norm)[amp_mask])#rms(np.log(amp_norm), pupil)
        strehl, strehl_phase, strehl_amp = get_strehl(phase, amp_norm, amp_mask)
        #strehl = np.exp(-phase_rms**2) * np.exp(-amp_lnrms**2)

        logger.info(f'Estimated phase RMS: {phase_rms:.3} (rad)')
        logger.info(f'Estimated amplitude RMS: {amp_rms*100:.3} (%)')
        logger.info(f'Estimated Strehl: {strehl:.2f} ({strehl_phase:.2f} phase-only and {strehl_amp:.2f} amplitude-only)')

        if update_shmim:
            update_estimate_shmims(phase * amp_mask, amp, config_params)

    if dmdivstream_in is None:
        dmdivstream.close()
    if camstream_in is None:
        camstream.close()
    #del camstream, dmdivstream

    if skip_estimation:
        return Imeas_cen
    else:
        return fitdict, Imeas_cen

def update_estimate_shmims(phase, amp, config_params):

    phase_shmim_name = config_params.get_param('estimation', 'phase_shmim', str)
    amp_shmim_name = config_params.get_param('estimation', 'amp_shmim', str)
    N = config_params.get_param('estimation', 'N', int)

    try:
        # open shmims here
        phasestream = ImageStream(phase_shmim_name, expected_shape=(N,N))
        ampstream = ImageStream(amp_shmim_name, expected_shape=(N,N))
    except RuntimeError:
        logger.info(f'Failed to open shmims {phase_shmim_name} and {amp_shmim_name}. Trying to create...')
        # assume this means they need to be created
        # and then try opening again
        create_shmim(phase_shmim_name, (N,N))
        create_shmim(amp_shmim_name, (N,N))
        phasestream = ImageStream(phase_shmim_name)
        ampstream = ImageStream(amp_shmim_name)

    phasestream.write(phase.astype(phasestream.buffer.dtype))
    ampstream.write(amp.astype(ampstream.buffer.dtype))
    logger.info(f'Updated shmims {phase_shmim_name} and {amp_shmim_name}')

    phasestream.close()
    ampstream.close()


# ------ CONFIGURATION -------

def replace_symlink(symfile, newfile):
    if path.islink(symfile) or path.exists(symfile):
        remove(symfile)
    symlink(newfile, symfile)
    logger.info(f'symlinked {newfile} to {symfile}')

class Configuration(configparser.ConfigParser):

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self._parse_config()

    def _parse_config(self):
        self.read(path.join('/opt/MagAOX/config', self.filename + '.conf'))

    def get_param(self, skey, key, dtype):
        val = self.get(skey, key) # still a str
        vallist = val.split(',') # handle lists
        if len(vallist) == 1:
            return dtype(vallist[0])
        else:
            return [dtype(v) for v in vallist]

    def update_from_dict(self, update_dict):
        for key, value in update_dict.items():
            section, option = key.split('.')
            try:
                self.get(section, option) # needed to verify that option already exists
                self.set(section, option, value=value)
            except configparser.NoSectionError as e:
                raise RuntimeError(f'Could not find section "{section}" in config file. Double-check override option "{key}={value}."') from e
            except configparser.NoOptionError as e:
                raise RuntimeError(f'Could not find option "{option}" in "{section}" section of config file. Double-check override option "{key}={value}."') from e

def rsync_calibration_directory(remote, config_params, dry_run=False):
    import os

    validate_calibration_directory(config_params)

    local_calibpath = config_params.get_param('calibration', 'path', str)
    remote_calibpath = remote + ':' + local_calibpath + '/'
    
    logger.info(f'Syncing {remote_calibpath} to {local_calibpath}.')

    cmdstr = 'rsync -azP ' + remote_calibpath + ' ' + local_calibpath
    if dry_run:
        cmdstr += ' --dry-run'

    os.system(cmdstr)

def update_symlinks_to_latest(config_params):

    validate_calibration_directory(config_params)
    calibpath = config_params.get_param('calibration', 'path', str)

    for curdir in SUBDIRS:
        filelist = sorted(glob(path.join(calibpath, curdir, '*.fits')))
        if len(filelist) > 0:
            latest = filelist[-1]
        else:
            continue
        sympath = path.join(calibpath, curdir+'.fits')
        replace_symlink(sympath, latest)


def validate_calibration_directory(config_params):
    '''
    Check that directory structure exists and is populated
    '''
    #check_and_make = lambda cpath: mkdir(cpath) if not path.exists(cpath) else 0
    check_and_make = lambda cpath: pathlib.Path(cpath).mkdir(parents=True, exist_ok=True) if not path.exists(cpath) else 0

    calibpath = config_params.get_param('calibration', 'path', str)
    check_and_make(calibpath)

    for curdir in SUBDIRS:
        check_and_make(path.join(calibpath, curdir))