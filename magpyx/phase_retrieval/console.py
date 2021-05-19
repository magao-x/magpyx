'''
To do:
* SYMLINKS AND DIRECTORY STRUCTURES
* preset (maybe can use existing code for this: fwtelsim, fwscind, stagescibs, fwsci2, camsci2 settings, ??)
* specify separate delay after dm diversity via indi vs dm cmd via shmims
* General way to override conf parameters
* closed loop
* function to set up directory structure from scratch
* nproc should be set from conf file
* npad should set the size of estrespM and everything that follows (reduce control matrix size)

helpers:
* parse config file???
* handle arguments (use config file if given, with override options; if not given, use defaults, with override options)
    - parse_known_args and then parse unkonwn args
    - unknown args update Configuration
    - which means Configuration needs to be defined by the console func to guarantee it'll be used with the overriden values everywhere
    - see https://stackoverflow.com/a/37367814 and https://stackoverflow.com/q/21920989
    - >> my_func --override camera.exp_time=3 diversity.navg=10

Configuration:

/opt/MagAOX/config
    - fdpr_dmncpc_camsci2.conf
    ...

In these files:

[camera]
name=camsci2
region_roi_x = ?
...
exp = ?
gain = ?

[instrument] [not sure about this - ugh] [could move these into a separate preset conf: preset_fdpr_dmncpc_camsci2.conf or something???]
fwpupil ?
fwscind ?
fwsci2 ?
stagescibs ?
fwtelsim ?

[diversity]
type=stage # or dm
values = [-0.3, -0.2, 0.2, 0.3] # [-60, -40, 40, 60] # microns RMS defocus or mm stage movement (deltas or abs?)
navg = 1
ndark = 100
dmdivchannel = dm02disp07  # maybe don't want this
dmModes = wooferModes # used if type=dm
camstage=stagesci2
port=7624

[estimation]
nzernike=45
npad=5
pupil=bump_mask
phase_shmim=fdpr_camsci2_phase
amp_shim=fdpr_camsci2_amp
nproc=3

[calibration]
path=/opt/MagAOX/calib/fdpr/dmncpc_camsci2
dmpath=/opt/MagAOX/calib/dm/alpao_260

[interaction]
hval = 0.05 # microns
Nact = 50
dm_map =
dm_mask = 

[control]
dmctrlchannel = dm02disp03 #??
nmodes=65
gain=0.5
leak=0.
niter=10
dmthresh=??
wfsthresh=??


Calibration:

(similar structure to cacao calibration products)
/opt/MagAOX/calib/fdpr
    - dmncpc_camsci2
        - dmmap.fits [sym]
        - dmmask.fits [sym]
        - controlM.fits [sym]
        - interM.fits [sym]
        - dmmap
            - dmmap_<date>.fits
            ...
        ...
        - interM
            - interM_<date>.fits
            ...

'''
from os import path, symlink, remove, mkdir
import argparse
import configparser
from datetime import datetime

from astropy.io import fits
import numpy as np

from purepyindi import INDIClient

from ..utils import ImageStream, create_shmim
from ..instrument import take_dark
from ..dm.dmutils import get_hadamard_modes, map_vector_to_square
from ..dm import control
from ..imutils import rms, write_to_fits

from .estimation import multiprocess_phase_retrieval, get_magaox_fitting_params
from .measurement import take_measurements_from_config

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

def console_close_loop():
    pass

def close_loop():
    pass

def console_compute_control_matrix():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--descriptor', '-d', type=str, default='', help='Descriptor to add to file name to help distinguish the outputs [NOT IMPLEMENTED].')
    #parser.add_argument('--nmodes', '-n', type=int, default=None, help='Number of modes to use in the control matrix / number SVD singular values to keep. [Defaults to value specified in conf file]')
    parser.add_argument('--override','-o', type=str, default=None, nargs='+', help='Space-delimited list of config parameters to override, set as section.option=value')

    args = parser.parse_args()

    # get configuration
    config_params = Configuration(args.config)
    # update with override args, if any
    if args.override is not None:
        override = parse_override_args(args.override)
        config_params.update_from_dict(override)

    compute_control_matrix(config_params, write=True)

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

def console_measure_response_matrix():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--descriptor', '-d', type=str, default='', help='Descriptor to add to file name to help distinguish this measurement.')
    args = parser.parse_args()

    # get configuration
    config_params = Configuration(args.config)

    calib_path = config_params.get_param('calibration', 'path', str)
    descr = args.descriptor
    date = datetime.now().strftime("%Y%m%d-%H%M%S")

    # do the thing
    imcube = measure_response_matrix(config_params)

    outpath = path.join(calib_path, 'measrespM', f'measrespM_{descr}_{date}.fits')
    if outpath is not None:
        fits.writeto(outpath, imcube)
        logger.info(f'Wrote interaction measurements to {outpath}')

    # replace symlink
    sympath = path.join(calib_path, 'measrespM.fits')
    replace_symlink(sympath, outpath)

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

def console_estimate_oneshot():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--shmims', type=bool, default=True, help='Update the phase and amplitude shmims? Default: True')
    parser.add_argument('--write', type=bool, default=False, help='Write the estimated phase and amplitude to file? Default: False')
    args = parser.parse_args()

    # get parameters from config file
    config_params = Configuration(args.config)

    # do the thing
    estimate_oneshot(config_params, update_shmim=args.shmims, write_out=args.write)


def estimate_oneshot(config_params, update_shmim=True, write_out=False):

    imcube = take_measurements_from_config(config_params, delay=config_params.get_param('diversity', 'delay', float))

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

def console_estimate_response_matrix():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--fits_file', type=str, default=None , help='Explicitly specify the fits file to process. If not given, defaults to latest interaction matrix in the calib directory.')
    parser.add_argument('--nproc', type=int, default=2, help='Number of processes to spawn. [Default: 2]')
    args = parser.parse_args()

    # get parameters from config file
    config_params = Configuration(args.config)
    calib_path = config_params.get_param('calibration', 'path', str)
    sympath_in = path.join(calib_path, 'measrespM.fits')

    # find and read in fits cube
    if args.fits_file is not None:
        pathname = args.fits_file
    else:
        pathname = sympath_in
    
    with fits.open(pathname) as f:
        image_cube = f[0].data

    logger.info(f'Performing estimation for {image_cube.shape} FITS file.')

    # get fitting params
    fitting_params = get_magaox_fitting_params(camera=config_params.get_param('camera', 'name', str),
                                               wfilter=config_params.get_param('diversity', 'wfilter', str),
                                               mask=config_params.get_param('estimation', 'pupil', str),
                                               N=config_params.get_param('estimation', 'N', int),
                                               divtype=config_params.get_param('diversity', 'type', str),
                                               divvals=config_params.get_param('diversity', 'values', float),
                                               npad=config_params.get_param('estimation', 'npad', int),
                                               nzernikes=config_params.get_param('estimation', 'nzernike', int),
    )

    estdict = estimate_response_matrix(image_cube, fitting_params)
    estrespM = np.asarray(estdict['phase']) * fitting_params['pupil_analytic']

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    outname = path.join(calib_path, 'estrespM', f'estrespM_{date}.fits')

    # write phases out to file
    fits.writeto(outname, estrespM)

    # replace symlink
    sympath_out = path.join(calib_path, 'estrespM.fits')
    replace_symlink(sympath_out, outname)

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