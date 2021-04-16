'''
What goes here?

* measure_interaction_matrix
* measure_interaction_matrix_console
* one-shot estimation (measure + estimate)
* estimate_interaction_matrix
* estimate_interaction_matrix_console
* compute_control_matrix
* compute_control_matrix_console
* closed_loop_correction
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
dmdivchannel = dm02disp07

[estimation]
nzernike=45
npad=5
pupil=bump_mask

[calibration]
path=/opt/MagAOX/calib/fdpr/dmncpc_camsci2

[interaction]
hval = 0.05 # microns
other estimation parameters??

[control]
dmctrlchannel = dm02disp03 #??
threshold=??
basis=??
gain=0.5
leak=0.
niter=10


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

from .estimation import multiprocess_phase_retrieval, get_magaox_fitting_params

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

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
    config_params = parse_config(args.config)
    calib_path = get_config(config_params, 'calibration', 'path', str)
    sympath = path.join(calib_path, 'measrespM.fits')

    # find and read in fits cube
    if args.fits_file is not None:
        pathname = args.fits_file
    else:
        pathname = sympath
    
    with fits.open(pathname) as f:
        image_cube = f[0].data

    logger.info(f'Performing estimation for {image_cube.shape} FITS file.')

    # get fitting params
    fitting_params = get_magaox_fitting_params(camera=get_config(config_params, 'camera', 'name', str),
                                               wfilter=get_config(config_params, 'diversity', 'wfilter', str),
                                               mask=get_config(config_params, 'estimation', 'pupil', str),
                                               N=get_config(config_params, 'estimation', 'N', int),
                                               divtype=get_config(config_params,  'diversity', 'type', str),
                                               divvals=get_config(config_params, 'diversity', 'values', float),
                                               npad=get_config(config_params, 'estimation', 'npad', int),
                                               nzernikes=get_config(config_params, 'estimation', 'nzernike', int),
    )

    estdict = estimate_response_matrix(image_cube, fitting_params)
    estrespM = np.asarray(estdict['phase']) * fitting_params['pupil_analytic']

    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    outname = path.join(calib_path, 'estrespM', f'estrespM_{date}.fits')

    # write phases out to file
    fits.writeto(outname, estrespM)

    # replace symlink
    replace_symlink(sympath, outname)

def parse_config(configname):
    config = configparser.ConfigParser()
    config.read(path.join('/opt/MagAOX/config', configname+'.conf'))
    return config

def get_config(config, skey, key, dtype):
    val = config.get(skey, key) # still a str
    vallist = val.split(',') # handle lists
    if len(vallist) == 1:
        return dtype(vallist[0])
    else:
        return [dtype(v) for v in vallist]

def replace_symlink(symfile, newfile):
    if path.exists(symfile):
        remove(symfile)
    symlink(newfile, symfile)


