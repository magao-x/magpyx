'''
To do:
* preset (maybe can use existing code for this: fwtelsim, fwscind, stagescibs, fwsci2, camsci2 settings, ??)
X report iteration number when closed loop
X move stage back to starting position when closed loop
* explore IF recon from hadamard modes (what if you overconstrain?)
* way to load  old config (update symlinks to point to old product)
* way to process old calib products
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
    parser.add_argument('--recompute', '-c', action='store_true', help='Force a recomputation of the control matrix. Must be provided if any override args change the control matrix.')
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
    if args.recompute:
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
    #print(override_args)
    #keyval_pairs = [x.strip() for x in override_args.split(' ')]
    argdict = {}
    for keyval in override_args:
        key, val = keyval.split('=')
        argdict[key] = val
    return argdict    
