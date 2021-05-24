'''
Tools to run FDPR, calibrate, and close the loop on MagAO-X
'''

from astropy.io import fits
import numpy as np
import configparser

from ..utils import ImageStream, create_shmim
from ..instrument import take_dark
from ..dm.dmutils import get_hadamard_modes, map_vector_to_square
from ..dm import control
from ..imutils import rms, write_to_fits

from purepyindi import INDIClient

from .estimation import multiprocess_phase_retrieval, get_magaox_fitting_params
from .measurement import take_measurements_from_config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

def measure_and_estimate_phase_vector(config_params, client, dmstream, camstream, darkim):
    '''
    wfsfunc for closed loop control: measure and return pupil-plane phase vector
    '''

    # take the measurement and run the estimator
    estdict = estimate_oneshot(config_params, update_shmim=True, write_out=False,
                               client=client, dmstream=dmstream, camstream=camstream,
                               darkim=darkim)

    # return the parameter of interest
    mask = config_params['fitting_region']
    phasevec = config_params['phase'][0][mask]

    # remove ptt first, I suppose
    return phasevec

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
    dmstream = ImageStream(config_params.get_param('diversity', 'dmdivchannel', str))
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

    # set up the wfs function
    wfsfuncdict = {
        'config_params' : config_params,
        'client' : client,
        'dmstream' : dmstream,
        'camstream' : camstream,
        'darkim' : darkim
    }
    wfsfunc =measure_and_estimate_phase_vector
    
    control.closed_loop(dmstream, ctrlmat, wfsfunc, niter=niter, gain=gain,
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
    npix = config_params.get_param('control', 'npix', int)

    # reduce measured data to npix region
    shape = hmeas.shape[-1] # I'm assuming it's square, as always...
    cen = shape//2
    slicezyx = (slice(None,None),
               slice(cen-npix//2,cen+npix//2),
               slice(cen-npix//2,cen+npix//2))
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

    # write these out to file
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
    imcube = take_measurements_from_config(config_params, dm_cmds=dm_cmds, delay=config_params.get_param('diversity', 'delay', float))

    return imcube.swapaxes(0,1)

def estimate_oneshot(config_params, update_shmim=True, write_out=False, client=None, dmstream=None, camstream=None, darkim=None):

    imcube = take_measurements_from_config(config_params,
                                           delay=config_params.get_param('diversity', 'delay', float),
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
    estdict = estimate_response_matrix([imcube,], fitting_params) # not sure if this is the function I want to use

    # report (maybe tell the user RMS, Strehl, etc. and tell them what shmims/files are updated)
    phase = estdict['phase'][0] * fitting_params['pupil_analytic']
    amp = estdict['amp'][0]

    phase_rms = rms(phase, fitting_params['pupil_analytic'].astype(bool)) # is pupil_analytic already boolean?
    amp_rms = rms(amp, fitting_params['pupil_analytic'].astype(bool)) # NOT NORMALIZED PROPERLY

    logger.info(f'Estimated phase RMS: {phase_rms}')
    logger.info(f'Estimated amplitude RMS (meaningless until normalized): {amp_rms}')
    #logger.info(f'Estimated Strehl: {strehl}')

    if update_shmim:
        update_estimate_shmims(phase, amp, config_params)

    return estdict

def update_estimate_shmims(phase, amp, config_params):

    phase_shmim_name = config_params.get_param('estimation', 'phase_shmim', str)
    amp_shmim_name = config_params.get_param('estimation', 'amp_shmim', str)
    N = config_params.get_param('estimation', 'N', int)

    try:
        # open shmims here
        phasestream = ImageStream(phase_shmim_name)
        ampstream = ImageStream(amp_shmim_name)
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

def estimate_response_matrix(image_cube, params, processes=2):
    # do all the processing
    rlist = multiprocess_phase_retrieval(image_cube, params, processes=processes)
    # turn list of dictionaries into dictionary of lists
    return {k: [cdict[k] for cdict in rlist] for k in rlist[0]}

def replace_symlink(symfile, newfile):
    if path.exists(symfile):
        remove(symfile)
    symlink(newfile, symfile)

def parse_override_args(override_args):
    '''
    Map key1=val1 key2=val2 into dictionary, I guess
    '''
    keyval_pairs = [x.strip() for x in override_args.split(' ')]
    argdict = {}
    for keyval in keyval_pairs:
        key, val = keyval.split('=')
        argdict[key] = val
    return argdict    

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
    pupil_coords_physical = get_coords(N, delta_pupil_phys)
    
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
        div_axial = np.asarray(divvals)
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
        'zbasis' : zbasis,
        'zkvals' : div_axial,
    }