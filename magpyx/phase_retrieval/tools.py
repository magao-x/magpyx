'''
Tools to run FDPR, calibrate, and close the loop on MagAO-X
'''
import configparser
from os import path, symlink, remove, mkdir
from datetime import datetime


from astropy.io import fits
import numpy as np
from skimage.filters.thresholding import threshold_otsu

from ..utils import ImageStream, create_shmim
from ..instrument import take_dark
from ..dm.dmutils import get_hadamard_modes, map_vector_to_square
from ..dm import control
from ..imutils import rms, write_to_fits, remove_plane

from purepyindi import INDIClient

from .estimation import multiprocess_phase_retrieval, get_coords, arbitrary_basis, gauss_convolve, defocus_rms_to_lateral_shift, downscale_local_mean, DEFAULT_STEPS, STEPS_NOXY
from .. import pupils
from .measurement import take_measurements_from_config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

def measure_and_estimate_phase_vector(config_params=None, client=None, dmstream=None, camstream=None, darkim=None, wfsmask=None):
    '''
    wfsfunc for closed loop control: measure and return pupil-plane phase vector
    '''
    # take the measurement and run the estimator
    estdict, fitting_params = estimate_oneshot(config_params, update_shmim=True, write_out=False,
                               client=client, dmstream=dmstream, camstream=camstream,
                               darkim=darkim)

    # remove ptt and return (I guess)
    # ugh, the wfsmask is defined within the fit region
    fitting_slice = fitting_params['fitting_slice']
    phase_pttrem = remove_plane(estdict['phase'][0][fitting_slice], wfsmask)
    return phase_pttrem[wfsmask] 

def measure_and_estimate_focal_field():
    '''
    wfsfunc for closed loop control: measure and return focal-plane E field vector

    Intended for EFC-like operations
    '''
    pass

def close_loop(config_params):

    # open indi client connection
    client = INDIClient('localhost', config_params.get_param('diversity', 'port', int))
    client.start()

    # open shmims
    dmstream = ImageStream(config_params.get_param('control', 'dmctrlchannel', str))
    camname = config_params.get_param('camera', 'name', str)
    camstream = ImageStream(camname)

    # take a dark (eventually replace this with the INDI dark [needs some kind of check to see if we have a dark, I guess])
    darkim = take_dark(camstream, client, camname, config_params.get_param('diversity', 'ndark', int))

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
        'config_params' : config_params,
        'client' : client,
        'dmstream' : dmstream,
        'camstream' : camstream,
        'darkim' : darkim,
        'wfsmask' : wfsmask
    }
    wfsfunc = measure_and_estimate_phase_vector
    
    control.closed_loop(dmstream, ctrlmat, wfsfunc, dm_map, dm_mask, niter=niter, gain=gain,
                        leak=leak, delay=delay, paramdict=wfsfuncdict)

    # close all the things
    client.stop()
    dmstream.close()
    camstream.close()

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
        dm_mask = f[0].data
    dmthresh = config_params.get_param('control', 'dmthreshold', float)
    wfsthresh = config_params.get_param('control', 'wfsthreshold', float)
    ninterp = config_params.get_param('control', 'ninterp', int)
    #npix = config_params.get_param('control', 'npix', int)

    # reduce measured data to npix region
    fitting_params = get_magaox_fitting_params(camera=config_params.get_param('camera', 'name', str),
                                            wfilter=config_params.get_param('diversity', 'wfilter', str),
                                            mask=config_params.get_param('estimation', 'pupil', str),
                                            N=config_params.get_param('estimation', 'N', int),
                                            divtype=config_params.get_param('diversity', 'type', str),
                                            divvals=config_params.get_param('diversity', 'values', float),
                                            npad=config_params.get_param('estimation', 'npad', int),
                                            nzernikes=config_params.get_param('estimation', 'nzernike', int))
    fitting_slice = fitting_params['fitting_slice']
    slicezyx = (slice(None,None),*fitting_slice)
    hmeas = hmeas[slicezyx]

    if nmodes is None:
        nmodes = config_params.get_param('control', 'nmodes', int)

    ctrldict = control.get_control_matrix_from_hadamard_measurements(hmeas,
                                                                     hmodes,
                                                                     hval,
                                                                     dm_map,
                                                                     dm_mask,
                                                                     wfsthresh=wfsthresh,
                                                                     dmthresh=dmthresh,
                                                                     ninterp=ninterp,
                                                                     nmodes=nmodes)

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

    # get the hadamard modes
    nact = config_params.get_param('interaction', 'nact', int)
    hval = config_params.get_param('interaction', 'hval', float)
    with fits.open(config_params.get_param('interaction', 'dm_map', str)) as f:
        dm_map = f[0].data
    with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
        dm_mask = f[0].data
    hmodes = get_hadamard_modes(nact)

    # reshape for DM
    hmodes_sq = np.asarray([map_vector_to_square(cmd, dm_map, dm_mask) for cmd in hmodes])
    # +/- and scaling
    dm_cmds = np.concatenate([hmodes_sq, -hmodes_sq]) * hval

    logger.info(f'Taking measurements for interaction matrix...')
    imcube = take_measurements_from_config(config_params, dm_cmds=dm_cmds)

    return imcube.swapaxes(0,1)

def estimate_oneshot(config_params, update_shmim=True, write_out=False, client=None, dmstream=None, camstream=None, darkim=None):

    imcube = take_measurements_from_config(config_params,
                                           client=client,
                                           dmstream=dmstream,
                                           camstream=camstream,
                                           darkim=darkim)

    # estimate    
    # get fitting params
    fitting_params = get_magaox_fitting_params(camera=config_params.get_param('camera', 'name', str),
                                               wfilter=config_params.get_param('diversity', 'wfilter', str),
                                               mask=config_params.get_param('estimation', 'pupil', str),
                                               N=config_params.get_param('estimation', 'N', int),
                                               divtype=config_params.get_param('diversity', 'type', str),
                                               divvals=config_params.get_param('diversity', 'values', float),
                                               npad=config_params.get_param('estimation', 'npad', int),
                                               nzernikes=config_params.get_param('estimation', 'nzernike', int))
    estdict = estimate_response_matrix([imcube,], # not sure if this is the function I want to use
                                       fitting_params,
                                       processes=config_params.get_param('estimation', 'nproc', int),
                                       gpus=config_params.get_param('estimation', 'gpus', int))     

    # report (maybe tell the user RMS, Strehl, etc. and tell them what shmims/files are updated)
    pupil = fitting_params['pupil_analytic'].astype(bool)
    amp = estdict['amp'][0]
    amp_norm = amp / np.mean(amp[pupil])

    # threshold phase based on amplitude? (reject phase values where amplitude is < some threshold)
    thresh_amp = threshold_otsu(amp_norm)
    threshold = config_params.get_param('control', 'ampthreshold', float)
    amp_mask = amp_norm > (thresh_amp*threshold)

    estdict['phase'][0] *= amp_mask
    phase = estdict['phase'][0] * pupil

    phase_rms = np.std(phase[pupil])#rms(phase, pupil)
    amp_rms = np.std(amp_norm[pupil])#rms(amp_norm, pupil)
    amp_lnrms = np.std(np.log(amp_norm)[pupil])#rms(np.log(amp_norm), pupil)
    strehl = get_strehl(phase, amp_norm, pupil)
    #strehl = np.exp(-phase_rms**2) * np.exp(-amp_lnrms**2)

    logger.info(f'Estimated phase RMS: {phase_rms:.3} (rad)')
    logger.info(f'Estimated amplitude RMS: {amp_rms*100:.3} (%)')
    logger.info(f'Estimated Strehl: {strehl}')

    if update_shmim:
        update_estimate_shmims(phase, amp, config_params)

    return estdict, fitting_params

def get_strehl(phase, amplitude, mask):
    
    Efield = amplitude * np.exp(1j*phase)
    Efield /= np.sqrt(np.sum(Efield * Efield.conj()))

    amp = np.abs(Efield)
    phase = np.angle(Efield)
    log_amp = np.log(amp)
    varlogamp = np.var(log_amp[mask])

    varphase = np.var(phase[mask])    
    return np.exp(-varphase) * np.exp(-varlogamp)


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

def estimate_response_matrix(image_cube, params, processes=2, gpus=None, fix_xy_to_first=False):
    '''
    A thin wrapper around estimation.multiprocess_phase_retrieval.

    If fix_xy_to_first==True, this will do the regular fit to the first set of images in the stack,
    and then fix that value for the remainder of the fits. Use case: tip/tilt is not orthogonal to the
    Hadamard modes, so you don't want to independently fit that and remove it from the phase solution.
    '''
    if fix_xy_to_first:
        logger.info('Fitting (xk, yk) from first image to fix for all other estimates.')
        r0 = multiprocess_phase_retrieval([image_cube[0],], params, processes=processes, gpus=gpus)
        xk = r0[0]['param_dict']['xk'][0]
        yk = r0[0]['param_dict']['yk'][0]
        steps = STEPS_NOXY
    else:
        steps = DEFAULT_STEPS
        xk = yk = None
    # do all the processing
    rlist = multiprocess_phase_retrieval(image_cube, params, processes=processes, gpus=gpus, steps=steps, xk_in=xk, yk_in=yk)
    # turn list of dictionaries into dictionary of lists
    return {k: [cdict[k] for cdict in rlist] for k in rlist[0]}

def replace_symlink(symfile, newfile):
    if path.exists(symfile):
        remove(symfile)
    symlink(newfile, symfile)

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

def get_magaox_fitting_params(camera='camsci2', wfilter='Halpha', mask='bump_mask', N=256, divtype='dm', divvals=[-0.3, -0.2, 0.2, 0.3], npad=5, nzernikes=45):
    '''
    Get the right set of parameters for each configuration.

    Parameters:
        divtype : str
            Either "dm" or "stage"
        divvals : list
            If divtype is "dm", then a list of defocus values send to the DM (um RMS). Ex: [-0.3, -0.2, 0.2, 0.3]
            If divtype is "stage", then a list of stage positions relative to nominal focus (mm). Ex: [-60, -40, 40, 60]
    '''
    
    #---This can't be the best way to do this.---
    
    # focal lengths
    fdict = {
        'camsci1' : 621e-3,
        'camsci2' : 621e-3,
    }
    # pupil sizes
    maskdict = {
        'bump_mask' : 8.604e-3,
        'open' : 9e-3
    }
    # center waves
    filterdict = {
        'Halpha' : 656.4228678034681e-9
    }
    # pixel sizes
    detdict = {
        'camsci1' : 13e-6,
        'camsci2' : 13e-6,
    }
    # pupil masks
    pupilfuncdict =  {
        'bump_mask' : pupils.get_coronagraphic_pupil,
        'open' : pupils.get_open_pupil,
    }
    # pupil rotation
    pupilrotdict = {
        'camsci1' : 53.5,
        'camsci2' : 53.5,
    }
    # pupil parity
    pupilparitydict = {
        'camsci1' : (slice(None,None,-1),slice(None,None,-1)),
        'camsci2' : (slice(None,None,-1),slice(None,None,-1))
    }
    
    #----------
    
    f = fdict.get(camera, None)
    D = maskdict.get(mask, None)
    wavelen = filterdict.get(wfilter, None)
    delta_focal = detdict.get(camera, None)
    pupilfunc = pupilfuncdict.get(mask, None)
    pupilrot = pupilrotdict.get(camera, None)
    pupilparity = pupilparitydict.get(camera, None)
    
    if f is None:
        raise ValueError(f'Could not find a focal length for camera {camera}.')
    if D is None:
        raise ValueError(f'Could not find a pupil size for pupil mask {mask}.')
    if wavelen is None:
        raise ValueError(f'Could not find a center wave for filter {wfilter}.')
    if delta_focal is None:
        raise ValueError(f'Could not find a pixel scale for camera {camera}.')
    if pupilfunc is None:
        raise ValueError(f'Could not find an analytic pupil for mask {mask}.')
    if pupilrot is None:
        raise ValueError(f'Could not find a pupil rotation for camera {camera}.')
    if pupilparity is None:
        raise ValueError(f'Could not find pupil parity for camera {camera}.')
      
    # f/#
    fnum = f/D
    
    # pupil and focal plane coordinates
    delta_pupil = 1/(N*delta_focal)
    delta_pupil_phys = delta_pupil*wavelen*f
    
    focal_coords = get_coords(N, delta_focal)
    pupil_coords = get_coords(N, delta_pupil, center=True)
    #pupil_coords_physical = get_coords(N, delta_pupil_phys)
    
    # analytic pupil mask
    pupil_analytic_upsampled = pupilfunc(delta_pupil_phys/10, N*10, extra_rot=pupilrot)[pupilparity]
    pupil_analytic = downscale_local_mean(pupil_analytic_upsampled, (10,10))
    pupil_binary = pupilfunc(delta_pupil_phys, N, extra_rot=pupilrot)[pupilparity].astype(bool)

    # fitting region
    idx, idy = np.indices((N,N))
    fitting_slice = (slice(idy[pupil_binary].min()-npad, idy[pupil_binary].max()+npad),
                     slice(idx[pupil_binary].min()-npad, idx[pupil_binary].max()+npad))
    fitting_region = np.zeros((N,N), dtype=bool)
    fitting_region[fitting_slice] = True
    
    # zernike basis
    zbasis = arbitrary_basis(fitting_region, nterms=nzernikes, outside=0)[3:] * gauss_convolve(pupil_analytic, 3)

    # defocus diversity
    if divtype.lower() == 'dm':
        div_axial = defocus_rms_to_lateral_shift(-np.asarray(divvals)*1e-6, fnum)
    elif divtype.lower() == 'stage':
        div_axial = np.asarray(divvals)*1e-3
    else:
        raise ValueError("divtype must be either 'dm' or 'stage'.")
    
    return {
        'f' : f,
        'D' : D,
        'fnum' : fnum,
        'wavelen' : wavelen,
        'focal_coords' : focal_coords,
        'pupil_coords' : pupil_coords,
        'pupil_analytic' : pupil_analytic,
        'fitting_region' : fitting_region,
        'fitting_slice' : fitting_slice,
        'zbasis' : zbasis,
        'zkvals' : div_axial,
    }

def rsync_calibration_directory(remote, config_params, dry_run=False):
    import os

    validate_calibration_directory(config_params)

    local_calibpath = config_params.get_param('calibration', 'path', str)
    remote_calibpath = remote + ':' + local_calibpath
    
    logger.info(f'Syncing {remote_calibpath} to {local_calibpath}.')

    cmdstr = 'rsync -azP ' + remote_calibpath + ' ' + local_calibpath
    if dry_run:
        cmdstr += ' --dry-run'

    os.system(cmdstr)

def validate_calibration_directory(config_params):
    '''
    Check that directory structure exists and is populated
    '''
    check_and_make = lambda cpath: mkdir(cpath) if not path.exists(cpath) else 0

    calibpath = config_params.get_param('calibration', 'path', str)
    check_and_make(calibpath)

    subdirs = ['ctrlmat', 'dmmap', 'dmmask', 'dmmodes', 'estrespM', 'ifmat',
               'measrespM', 'singvals', 'wfsmap', 'wfsmask', 'wfsmodes']

    for curdir in subdirs:
        check_and_make(path.join(calibpath, curdir))

