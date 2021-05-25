'''
To do:
* SYMLINKS AND DIRECTORY STRUCTURES
* preset (maybe can use existing code for this: fwtelsim, fwscind, stagescibs, fwsci2, camsci2 settings, ??)
* specify separate delay after dm diversity via indi vs dm cmd via shmims
X General way to override conf parameters
X closed loop
X function to set up directory structure from scratch
X npad should set the size of estrespM and everything that follows (reduce control matrix size) (or from fitting region??? yes. how is that specified?)
X report Strehl
X fix amplitude normalization

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
from os import path
import argparse
from datetime import datetime

from astropy.io import fits
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

from .tools import (close_loop, compute_control_matrix, measure_response_matrix, estimate_oneshot,
                    validate_calibration_directory, Configuration, replace_symlink, estimate_response_matrix,
                    get_magaox_fitting_params)

def console_close_loop():
    #argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--skip_ctrl', '-s', type=bool, action='store_true', help='By default, the control matrix is always recomputed right before closing the loop. This flag skips this step.')
    parser.add_argument('--override','-o', type=str, default=None, nargs='+', help='Space-delimited list of config parameters to override, set as section.option=value')
    args = parser.parse_args()

    # get configuration
    config_params = Configuration(args.config)
    # update with override args, if any
    if args.override is not None:
        override = parse_override_args(args.override)
        config_params.update_from_dict(override)

    validate_calibration_directory(config_params)

    # recompute control matrix
    if not args.skip_ctrl:
        compute_control_matrix(config_params)

    # close the loop
    close_loop(config_params)

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

    validate_calibration_directory(config_params)

    compute_control_matrix(config_params, write=True)

def console_measure_response_matrix():
    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--descriptor', '-d', type=str, default='', help='Descriptor to add to file name to help distinguish this measurement.')
    parser.add_argument('--override','-o', type=str, default=None, nargs='+', help='Space-delimited list of config parameters to override, set as section.option=value')
    args = parser.parse_args()

    # get configuration
    config_params = Configuration(args.config)
    # update with override args, if any
    if args.override is not None:
        override = parse_override_args(args.override)
        config_params.update_from_dict(override)

    validate_calibration_directory(config_params)

    calib_path = config_params.get_param('calibration', 'path', str)
    descr = args.descriptor
    date = datetime.now().strftime("%Y%m%d-%H%M%S")

    # do the thing
    imcube = measure_response_matrix(config_params)

    outpath = path.join(calib_path, 'measrespM', f'measrespM_{descr}_{date}.fits')
    fits.writeto(outpath, imcube)
    logger.info(f'Wrote interaction measurements to {outpath}')

    # replace symlink
    sympath = path.join(calib_path, 'measrespM.fits')
    replace_symlink(sympath, outpath)

def console_estimate_oneshot():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--shmims', type=bool, default=True, help='Update the phase and amplitude shmims? Default: True')
    parser.add_argument('--write', type=bool, default=False, help='Write the estimated phase and amplitude to file? Default: False')
    parser.add_argument('--override','-o', type=str, default=None, nargs='+', help='Space-delimited list of config parameters to override, set as section.option=value')
    args = parser.parse_args()

    # get parameters from config file
    config_params = Configuration(args.config)
    # update with override args, if any
    if args.override is not None:
        override = parse_override_args(args.override)
        config_params.update_from_dict(override)

    validate_calibration_directory(config_params)

    # do the thing
    estimate_oneshot(config_params, update_shmim=args.shmims, write_out=args.write)

def console_estimate_response_matrix():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Name of the configuration file to use (don\'t include the path or extension).')
    parser.add_argument('--fits_file', type=str, default=None , help='Explicitly specify the fits file to process. If not given, defaults to latest interaction matrix in the calib directory.')
    parser.add_argument('--nproc', type=int, default=2, help='Number of processes to spawn. [Default: 2]')
    parser.add_argument('--override','-o', type=str, default=None, nargs='+', help='Space-delimited list of config parameters to override, set as section.option=value')
    args = parser.parse_args()

    # get parameters from config file
    config_params = Configuration(args.config)
    # update with override args, if any
    if args.override is not None:
        override = parse_override_args(args.override)
        config_params.update_from_dict(override)

    validate_calibration_directory(config_params)
    
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
