'''

To do:
* create measure_interaction matrix functions
* finish one-shot
    * need to finish reporting code
    * figure out general way to override conf parameters (for all console functions)
    * test
* create compute_control_matrix
    * well, write the thing
    * work out how to define a custom control basis
    * pre-define a couple control bases or something (IFs, zernikes, KL modes, fourier modes, ??)
    * write dm map, dm mask funcs and write to file
* create closed_loop_coorection
    * write it

What goes here?

X measure_interaction_matrix
X measure_interaction_matrix_console
X one-shot estimation (measure + estimate)
    - should add option to override diversity type and diversity values, etc.
    - I need a more general way over overriding config values that can be used across all console functions
    - Add RMS / Strehl reporting
X estimate_interaction_matrix
    - should this function take the difference of the +/- measurements and normalize properly? (divide by input value and 2)
    - input value (mag of hadamard modes) should be in the .conf file
X estimate_interaction_matrix_console
* compute_control_matrix
    - need to build interpolation outside beam footprint into DM modes / ctrl matrix
    - maybe from the SVD save out U, S, V as well as ctrl matrix (note: I think I'm using a different def of this than cacao)
* compute_control_matrix_console
* closed_loop_correction
    - I think this should accept an nmodes arg and recompute the ctrl matrix on the fly (easy if you've saved out U S V)
* closed_loop_correction_console

helpers:
* parse config file???
* handle arguments (use config file if given, with override options; if not given, use defaults, with override options)

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
from os import path, symlink, remove
import argparse
import configparser
from datetime import datetime

from astropy.io import fits
import numpy as np

from purepyindi import INDIClient

from ..utils import ImageStream
from ..instrument import take_dark
from ..dm.dmutils import get_hadamard_modes
from ..dm import control

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
    parser.add_argument('--nmodes', '-n', type=int, default=None, help='Number of modes to use in the control matrix / number SVD singular values to keep. [Defaults to value specified in conf file]')

    args = parser.parse_args()

    # get configuration
    config_params = Configuration(args.config)

    compute_control_matrix(config_params, nomdes=args.nmodes, write=True)

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
    hmeas_path = path.join(calib_path, 'measrepM.fits')
    with fits.open(hmeas_path) as f:
        hmeas = f[0].data
    nact = config_params.get_param('interaction', 'nact', int)
    hmodes = get_hadamard_modes(nact)
    hval = config_params.get_param('interaction', 'hval', float)

    # get dm map and mask and thresholds
    dm_calib_path = config_params.get_param('calibration', 'dmpath', str)
    with fits.open(path.join(dm_calib_path, 'dm_map.fits')) as f:
        dm_map = f[0].data
    with fits.open(path.join(dm_calib_path, 'dm_mask.fits')) as f:
        dm_mask = f[0].data
    dmthresh = config_params.get_param('control', 'dmthreshold', float)
    wfsthresh = config_params.get_param('control', 'wfsthreshold', float)
    ninterp = config_params.get_param('control', 'ninterp', int)

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
            outname = path.join(calib_path, key, f'{key}_{date}.fits')
            fits.write(outname, imcube)
            logger.info(f'Wrote out {outname}')
    
    # update all symlinks

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
    measure_response_matrix(config_params, outpath=path.join(calib_path, f'measrespM_{descr}_{date}.fits'))

def measure_response_matrix(config_params, outpath=None):

    # get the hadamard modes
    nact = config_params.get_param('interaction', 'nact', int)
    hval = config_params.get_param('interaction', 'hval', float)
    with fits.open(config_params.get_param('interaction', 'dm_map', float)) as f:
        dm_map = f[0].data
    with fits.open(config_params.get_param('interaction', 'dm_mask', float)) as f:
        dm_mask = f[0].data
    hmodes = get_hadamard_modes(nact)

    # reshape for DM
    hmodes_sq = np.asarray([map_vector_to_square(cmd, dm_map, dm_mask) for cmd in hmodes])
    # +/- and scaling
    dm_cmds = np.concatenate([hmodes_sq, -hmodes_sq]) * hval

    logger.info(f'Taking measurements for interaction matrix...')
    imcube = take_measurements_from_config(config_params, dm_cmds=dm_cmds)

    if outpath is not None:
        fits.write(outpath, imcube)
        logger.info(f'Wrote interaction measurements to {outpath}')

    return imcube

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

    imcube = take_measurements_from_config(config_params)

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
    estdict = estimate_response_matrix([imcube,], params) # not sure if this is the function I want to use

    # report (maybe tell the user RMS, Strehl, etc. and tell them what shmims/files are updated)
    phase = estdict['phase'][0] #* fitting_params['pupil_analytic']
    amp = estdict['amp'][0]

    phase_rms = imutils.rms(phase, fitting_params['pupil_analytic']) # is pupil_analytic a boolean?
    amp_rms = imutils.rms(amp, fitting_params['pupil_analytic']) # NOT NORMALIZED PROPERLY

    logger.info(f'Estimated phase RMS: {phase_rms}')
    logger.info(f'Estimated amplitude RMS (meaningless until normalized): {amp_rms}')
    #logger.info(f'Estimated Strehl: {strehl}')

    if update_shmim:
        update_estimate_shmims(phase, amp, config_params)

    # clean up (close shmims, etc.)
    client.close()
    dmstream.close()
    camstream.close()

    return estdict

def update_estimate_shmims(phase, amp, config_params):

    phase_shmim_name = config_params.get_param('estimation', 'phase_shmim', str)
    amp_shmim_name = config_params.get_param('estimation', 'amp_shmim', str)
    N = config_params.get_param('camera', 'region_roi_x', int) # I'm assuming ROI is always square

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
    ampstream.write(amp.astype(amp.buffer.dtype))
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
    sympath = path.join(calib_path, 'estrespM.fits')

    # find and read in fits cube
    if args.fits_file is not None:
        pathname = args.fits_file
    else:
        pathname = sympath
    
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
    replace_symlink(sympath, outname)

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