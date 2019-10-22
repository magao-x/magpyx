'''
To do:
* Clean all this up
* Make sure everything's still working as intended
* Develop notebooks with simple use case
* Write pyramid metric(s)

'''

from copy import deepcopy
from time import sleep
from random import shuffle

import numpy as np

import purepyindi as indi

from astropy.stats import sigma_clipped_stats
from astropy.io import fits

from scipy.optimize import leastsq, minimize_scalar, dual_annealing
from scipy.ndimage import center_of_mass
from scipy import stats
from scipy.special import jv

from skimage import draw

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('eye_doctor')

from ..utils import ImageStream

#-----purepyindi interaction-----

def get_value(client, device, prop, elem, wait=False, timeout=None):
    '''
    Helper function to return an INDI element
    
    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
        prop : str
            Property name
        elem : str
            Element name
    
    Returns: element value
    '''
    return 1.0 # FIX ME

    if wait:
        client.wait_for_properties([f'{device}.{prop}',], timeout=timeout)
    return client[f'{device}.{prop}.{elem}']

def send_value(client, device, prop, elem, value, wait=False, timeout=None):
    '''
    Helper function for setting an INDI element


    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
        prop : str
            Property name
        elem : str
            Element name
        value : any
            Value to set

    Returns: nothing
    
    '''
    if wait:
        client.wait_for_properties([f'{device}.{prop}',], timeout=timeout)
    client[f'{device}.{prop}.{elem}'] = value
        
def zero_dm(client, device,):
    '''
    Set all mode amplitudes to 0.

    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
    '''
    nmodes = len(client.devices[device].properties[prop].elements)
    send_modes_and_wait(client, device, {m:0 for m in range(nmodes)})

def send_modes_and_wait(client, device, mode_targ_dict, tol=1e-3, wait_for_properties=True, timeout=10):
    status_dict = {}
    for mode, targ in mode_targ_dict.items():
        status_dict.update({
            f'{device}.current_amps.{mode:0>2}': {
                'value': targ,
                'test': lambda current, value, tolerance=tol: abs(current - value) < tolerance,
            },
            f'{device}.target_amps.{mode:0>2}': {
                'value': targ,
                'test': lambda current, value, tolerance=tol: abs(current - value) < tolerance,
            }})
    return # FIX ME
    #return client.wait_for_state(status_dict, wait_for_properties=wait_for_properties, timeout=timeout)

#----- metrics and analysis -----

def gaussfit(image, clipping=None):
    '''Fit a Gaussian'''
    cenyx = np.where(image == image.max())
    peak = image[cenyx][0]
    
    if clipping is not None:
        y = int(np.rint(cenyx[0][0]))
        x = int(np.rint(cenyx[1][0]))
        im = image[y-clipping//2:y+clipping//2, x-clipping//2:x+clipping//2]
    else:
        im = image
    
    shape = im.shape
    fwhm = 10.
    init = [fwhm, peak, cenyx[0][0], cenyx[1][0]]
    return leastsq(gausserr, init, args=(shape, im))
    
def gausserr(params, shape, image):
    '''Gauss error function'''
    fwhm, peak, ceny, cenx = params
    return gauss2d(fwhm, peak, (ceny, cenx), shape).flatten() - image.flatten()

def gauss_centroid(image, fwhm, clipping=None):
    '''
    Least squares fit a gaussian centroid. Less general than gaussfit
    '''
    cenyx = np.where(image == image.max())
    
    if clipping is not None:
        y = int(np.rint(cenyx[0][0]))
        x = int(np.rint(cenyx[1][0]))
        im = image[y-clipping//2:y+clipping//2, x-clipping//2:x+clipping//2]
    else:
        im = deepcopy(image)
        
    shape = im.shape
    init = [cenyx[0][0], cenyx[1][0]]
    return leastsq(gauss_centroid_err, init, args=(shape, im, fwhm), ftol=1e-5, xtol=1e-5)
    
def gauss_centroid_err(params, shape, image, fwhm):
    '''
    Gauss centroid error functions
    '''

    ceny, cenx = params
    #print(ceny, cenx)
    fitgauss = gauss2d(fwhm, (ceny, cenx), shape)
    
    return ((fitgauss - image)).flatten() # weight

def least_squares(image, model, weight=None):
    if weight is None:
        return np.sum((image - model)**2)
    else:
        return np.sum((image - model)**2 * weight)

def gauss2d(fwhm, center, size):
    """
    Generate a 2D Gaussian

    Parameters:
        fwhm : float
            FWHM of Gaussian. Same value is used for both dimensions
        center : tuple of floats
            Center of Gaussian (can be subpixel): (y, x)
        size : tuple of ints
            Shape of array to generate Gaussian

    Returns: 2D array with the Gaussian
    """
    y = np.arange(0, size[0])[:,None]
    x = np.arange(0, size[1])
    y0 = center[0]
    x0 = center[1]
    
    sigma = 2 * np.sqrt(2 * np.log(2) ) * fwhm
    
    return 1./ ( 2 * np.pi * sigma**2) * np.exp( - ((x-x0)**2 + (y-y0)**2) / (2 * sigma**2))

def subtract_bg(image, stype=0):
    '''
    Subtract the background from an image in various ways
    '''
    if stype == 0:
        # full image median
        return image - np.median(image)
    elif stype == 1:
        # edge median
        edge_mask = np.zeros(image.shape, dtype=bool)
        edge_mask[:5] = 1.
        edge_mask[:,:5] = 1.
        edge_mask[-5:] = 1.
        edge_mask[:,-5:] = 1.
        return image - np.median(image[edge_mask])
    elif stype == 2:
        mode, _ = stats.mode(image, axis=None)
        return image - mode
    elif stype == 3:
        # row by row and then column by column mode subtraction
        imsub = deepcopy(image)
        m1 = np.median(imsub, axis=0)#stats.mode(imsub, axis=0)
        imsub -= m1[None,:]
        m2 = np.median(imsub, axis=1)#stats.mode(imsub, axis=1)
        imsub -= m2[:,None]
        # remove global median
        return imsub - np.median(imsub)

def subtract_bg_median_sigmaclip(image, sigma):
    '''
    Subtract a sigma-clipped median background
    '''
    im = deepcopy(image)
    _, median1, _ = sigma_clipped_stats(im, sigma=sigma, axis=0, cenfunc='median')
    im -= median1[None,:]
    _, median2, _ = sigma_clipped_stats(im, sigma=sigma, axis=1, cenfunc='median')
    im -= median2[:,None]
    return im

def find_peak(image, stype=0, clipping=None):
    '''
    Extract the peak from an image by either a
    naive maximum or a Gaussian fit
    '''
    if stype == 0:
        # extract peak pixel
        return image.max()
    else:
        # fit gaussian
        params = gaussfit(image, clipping)
        return params[0][1]

def obscured_airy_disk(I0, wavelength, fnum, pixscale, cenyx, shape):
    '''
    Generate an obscured airy pattern for a 29% obscured pupil
    '''
    eta = 0.29
    
    indices = np.indices(shape)
    r = np.sqrt( (indices[0]-cenyx[0])**2 + (indices[1]-cenyx[1])**2)
    arg = r * np.pi / (wavelength * fnum) * pixscale
    arg[arg == 0] = 1e-16
    
    #coeff = I0 / (1 - eta**2)**2
    t1 = 2 * jv(1, arg) / arg 
    t2 = 2*eta*jv(1, eta*arg) / arg
    
    airy =  I0 * (t1 - t2)**2 / np.sqrt(1-eta)
    #airy[np.isnan(airy)] = I0 * (1-eta)# handle central pixel
    
    return airy

def get_image_peak(arrlist):
    peaks = []
    for image in arrlist:
        im_bgsub = subtract_bg(image, stype=1)
        im_peak = find_peak(im_bgsub, stype=0)
        peaks.append(im_peak)
    avgpeak = np.mean(peaks)
    return avgpeak

def get_image_coresum(arrlist, radius=10):
    
    # background subtract and then average
    coresum = []
    for image in arrlist:
        im_bgsub = subtract_bg(image, stype=1)
        #ims_bgsub.append(im_bgsub)
        #avg = np.mean(ims_bgsub, axis=0)

        # two step centroid: plop down a mask of 2*radius and then
        # center of mass to refine the centroid
        radius2 = 2*radius
        ceny, cenx = np.where(im_bgsub == im_bgsub.max())
        circ_centroid = draw.circle(ceny[0], cenx[0], radius2, im_bgsub.shape)
        circmask_centroid = np.zeros(im_bgsub.shape, dtype=bool)
        circmask_centroid[circ_centroid] = 1
        y, x = center_of_mass(im_bgsub * circmask_centroid)

        # core mask
        circ1 = draw.circle(y, x, radius, im_bgsub.shape)
        circmask = np.zeros(im_bgsub.shape, dtype=bool)
        circmask[circ1] = 1

        # annulus
        circ2 = draw.circle(y, x, radius2, im_bgsub.shape)
        annulus_mask = np.zeros(im_bgsub.shape, dtype=bool)
        annulus_mask[circ2] = 1
        annulus_mask[circ1] = 0

        # metric = sum(annulus) / sum(core)
        im_core = np.sum(im_bgsub[circmask])
        coresum.append(im_core)
        
    avgcore = np.mean(coresum)
    return -avgcore

def get_image_core_ring_ratio(arrlist, radius1=10, radius2=20):
    # alternate approach: take ratios of images and then average

    
    # background subtract and then average
    ratios = []
    for image in arrlist:
        im_bgsub = subtract_bg(image, stype=1)
        #ims_bgsub.append(im_bgsub)
        #avg = np.mean(ims_bgsub, axis=0)

        # two step centroid: plop down a mask of radius2 and then
        # center of mass to refine the centroid
        ceny, cenx = np.where(im_bgsub == im_bgsub.max())
        circ_centroid = draw.circle(ceny[0], cenx[0], radius2, im_bgsub.shape)
        circmask_centroid = np.zeros(im_bgsub.shape, dtype=bool)
        circmask_centroid[circ_centroid] = 1
        y, x = center_of_mass(im_bgsub * circmask_centroid)

        # core mask
        circ1 = draw.circle(y, x, radius1, im_bgsub.shape)
        circmask = np.zeros(im_bgsub.shape, dtype=bool)
        circmask[circ1] = 1

        # annulus
        circ2 = draw.circle(y, x, radius2, im_bgsub.shape)
        annulus_mask = np.zeros(im_bgsub.shape, dtype=bool)
        annulus_mask[circ2] = 1
        annulus_mask[circ1] = 0

        # metric = sum(annulus) / sum(core)
        im_core = np.sum(im_bgsub[circmask])
        im_annulus = np.sum(im_bgsub[annulus_mask])
        ratio = im_annulus / im_core
        if np.isinf(ratio):
            ratio = 999
        if np.isnan(ratio):
            ratio = 999
        ratios.append(ratio)
    avgratio = np.nanmean(ratios) # there shouldn't be nans
    return avgratio#, avg, circmask, annulus_mask

def fit_airy_disk(psf, wavelength, fnum, pixscale, cutout=100):
    
    # find centroid and cut out a subarray for fitting
    y, x = np.where(psf == psf.max())
    lower = lambda x: x if x > 0 else 0
    measured = psf[lower(y[0]-cutout//2):y[0]+cutout//2, lower(x[0]-cutout//2):x[0]+cutout//2]
    
    # parameters for airy disk fit
    shape = measured.shape
    ceny, cenx = np.where(measured == measured.max())
    bg = np.median(psf)
    # need to work on normalization. So many approaches favor driving the max to 0...
    psfmax = measured.max()
    
    res, _ = leastsq(airy_err, [ceny[0], cenx[0], bg], args=(measured, shape, psfmax, wavelength, fnum, pixscale),
            ftol=1e-5, xtol=1e-5)
    return res, measured
    
def airy_err(params, measured, shape, psfmax, wavelength, fnum, pixscale):
    # fit quantities
    ceny, cenx, bg = params
    airy = obscured_airy_disk(psfmax, wavelength, fnum, pixscale, (ceny, cenx), shape) + bg
    return (airy - measured).flatten()

def airy_metric(measured, model, penalty=0.):
    # consider adding penalty for low energy solutions
    #print(np.sqrt(np.sum((measured-model)**2)), penalty/np.sqrt(np.sum(measured**2)))
    return np.sqrt(np.sum((measured-model)**2)) + penalty/np.sqrt(np.sum(measured**2))

def get_pupil_variance(shmim, nimages, pupil_mask):
    images = grab_images(shmim, nimages)
    return np.var(np.mean(images, axis=0)[pupil_mask])

#-----the eye doctor-----

def move_measure_metric(val, client, device, shmim, nmode, nimages, metric, metric_dict):
    '''
    Move the DM, take a measurement, and return the value of the metric.
    '''

    # move
    send_modes_and_wait(client, device, {f'{nmode:0>2}' : val})
    #sleep(0.1)
    # measure
    images = shmim.grab_many(nimages)
    # metric
    return metric(images, **metric_dict)

def optimize_strehl(client, device, shmim, nmode, nimages, bounds, metric, metric_dict={}, tol=1e-5):
    res = minimize_scalar(move_measure_metric, bounds=bounds,
                          args=(client, device, shmim, nmode, nimages, metric, metric_dict),
                          method='bounded', options={'maxiter' : 100, 'xatol' : tol})
    return res['x']

def grid_sweep(client, device, shmim, n, nimages, curbounds, nsteps, nrepeats, metric, metric_dict, search_dict={}, debug=False):
    
    steps = np.linspace(curbounds[0], curbounds[1], num=nsteps, endpoint=True)
    
    curves = np.zeros((nrepeats, nsteps))
    for i in range( nrepeats):
        for j, s in enumerate(steps):
            # move
            send_modes_and_wait(client, device, {f'{n:0>2}' : s})
            #sleep(0.1)
            #measure
            images = np.zeros((1,50,50)) #shmim.grab_many(nimages) FIX ME
            #metric
            curves[i, j] = metric(images, **metric_dict)

    # skip processing and just pass the metric values back
    if debug:
        return steps, curves

    skind = search_dict.get('kind', 'fit')

    if skind == 'mean':
        # get the mean min
        return np.mean(steps[np.argmin(curves,axis=1)])
    elif skind == 'fit':
        # fit a quadratic
        # the problem here is that the fit could fail
        # end return an undesirable (read: very large) value
        
        # combine all sweeps into one dataset to fit
        c, b, a = np.polyfit( np.repeat(steps, nrepeats), curves.T.flatten(), deg=2)
        minima =  - b / (2 * c)
        mean = minima
        
        if (mean < curbounds[0]) or (mean > curbounds[1]):
            logger.warning('Bad quadratic fit!')
            return np.nan
        else:
            return mean
    else:
        raise ValueError('kind must be "mean" or "fit"!')

def eye_doctor(client, device, shmim, nimages, modes, bounds, search_kind='grid', search_dict={}, metric=get_image_coresum, metric_dict={},
               curr_prop='current_amps', targ_prop='target_amps', baseline=True):
    '''

    Predefined metrics:
    get_image_coresum
    peak
    core
    ratio
    pyramid

    Metric functions:
    func(imagecube, kw_arg1=sdf, kw_arg2=sdf)

    '''

    optimized_amps = []
    metric_vals = []
    # loop over modes
    for i, n in enumerate(modes):

        # baseline centers the search or sweep around the current value 
        if baseline:
            baseval = get_value(client, device, curr_prop, n)
            curbounds = baseval - np.asarray(bounds)
        else:
            baseval = 0.
            curbounds = bounds

        # measure
        meas_init = np.zeros((1,50,50)) #shmim.grab_many(nimages) FIX ME
        # metric
        val_init = metric(meas_init, **metric_dict)

        # grid_sweep or optimize on metric
        # measurements outside of metric
        if search_kind == 'minimize':
            tol = search_dict.get('tol', 1e-5)
            optval = optimize_strehl(client, device, n, nimages, curbounds, metric, metric_dict=metric_dict, tol=tol)
        elif search_kind == 'grid':
            nsteps = search_dict.get('nsteps', 20)
            nrepeats = search_dict.get('nrepeats', 3)
            optval = grid_sweep(client, device, shmim, n, nimages, curbounds,
                                nsteps, nrepeats, metric, metric_dict=metric_dict)
        else:
            raise ValueError('search_kind must be either "minimize" or "grid".')

        # send to optimized value
        send_modes_and_wait(client, device, {f'{n:0>2}' : optval})
        #sleep(0.1)

        # measure
        meas_final = np.zeros((1,50,50)) # FIX ME #shmim.grab_many(nimages)
        # metric
        val_final = metric(meas_final, **metric_dict)

        logger.info(f'Mode {n}: Optimized mode coefficient from {baseval:.2} to {optval:.2} microns')
        logger.info(f'Mode {n}: Optimized metric {metric} from {val_init:.2} to {val_final:.2}')

def build_sequence(client, device, shmim, nimages, metric=get_image_coresum, metric_dict={}, search_kind='grid', search_dict={}, curr_prop=None, targ_prop=None,
                   modes=range(2,36), ncluster=5, nrepeat=3, nseqrepeat=2, randomize=True, baseline=True, bounds=[-5e-3, 5e-3]):
    modes = list(modes)
    nmodes = len(modes)

    nfullgroups, mpartial = np.divmod(nmodes, ncluster)

    mode_args = []

    for j in range(nseqrepeat):
        # do full groups
        for m in range(nfullgroups):
            for k in range(nrepeat):
                curmodes = deepcopy(modes[1+m*ncluster:1+m*ncluster+ncluster])
                if randomize:
                    shuffle(curmodes)
                mode_args.append(curmodes)

        # do partial group
        if mpartial > 0:
            for k in range(nrepeat):
                curmodes = deepcopy(modes[-mpartial:])
                if randomize:
                    shuffle(curmodes)
                mode_args.append(curmodes)
                
    args = []
    for m in mode_args:
        args.append((client, device, shmim, nimages, m, bounds, search_kind, search_dict, metric, metric_dict, curr_prop, targ_prop, baseline,))
            
    return args

def eye_doctor_comprehensive(client, device, shmim, nimages, metric=get_image_coresum, metric_dict={}, search_kind='grid', search_dict={}, curr_prop=None, targ_prop=None,
                             modes=range(2,36), ncluster=5, nrepeat=3, nseqrepeat=2, randomize=True, baseline=True, bounds=[-5e-3, 5e-3]):

    # build sequence
    args_seq = build_sequence(client, device, shmim, nimages, metric=metric, metric_dict=metric_dict, search_kind=search_kind,
                              search_dict=search_dict, curr_prop=curr_prop, targ_prop=targ_prop, modes=modes, ncluster=ncluster,
                              nrepeat=nrepeat, nseqrepeat=nseqrepeat, randomize=randomize, baseline=baseline, bounds=bounds)

    # if focus is requested, do it first (it gets repeated later but oh well)
    if 2 in modes:
        logger.info('Optimizing focus first!')
        eye_doctor(client, device, shmim, nimages, [2,], bounds, search_kind=search_kind, search_dict=search_dict, metric=metric,
                   metric_dict=metric_dict, curr_prop=curr_prop, targ_prop=targ_prop, baseline=baseline)

    # do the eye doctor
    for args in args_seq:
        eye_doctor(*args)

def console_modal():

    # parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('portINDI', type=int, default=7624, help='INDI Port')
    parser.add_argument('device', type=str, help='INDI device with dm modes. [wooferModes, ncpcModes, tweeterModes]')
    parser.add_argument('shmim', type=str, help='Shared memory image [camsci1, camsci2, camlowfs, etc.]')
    parser.add_argument('core', type=float, help='Radius of the PSF core to measure')
    parser.add_argument('mode', type=int, help='Mode to optimize')
    parser.add_argument('range', type=float, help='Range of values in microns over which to perform a grid search.')
    parser.add_argument('--nsteps', type=int, default=20, help='Number of points to sample in grid search [Default: 20]')
    parser.add_argument('--nrepeats', type=int, default=3, help='Number of sweeps [Default: 3]')
    parser.add_argument('--nimages', type=int, default=1, help='Number of images to collect from shmim [Default: 1]')
    parser.add_argument('--reset',  action='store_true', help='Ignore the current value of the mode and optimize about 0')

    args = parser.parse_args()

    # connect to INDI server
    client = indi.INDIClient('localhost', args.portINDI)
    client.start()

    # connect to shmim
    shmim = ImageStream(args.shmim)

    # run eye doctor
    eye_doctor(client, args.device, shmim, args.nimages, [args.mode,], [-args.range/2., args.range/2], search_kind='grid',
               search_dict={'nsteps' : args.nsteps, 'nrepeats' : args.nrepeats},
               metric=get_image_coresum, metric_dict={'radius' : args.core},
               curr_prop='current_amps', targ_prop='target_amps', baseline=~args.reset)

def console_comprehensive():
    '''
    Comprehensive eye doctoring.

    Given a set of modes, split them into clusters, and iterate over the clusters,
    optimizing modes in a shuffled order. 
    '''
    
    # parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('portINDI', type=int, default=7624, help='INDI Port')
    parser.add_argument('device', type=str, help='INDI device with dm modes. [wooferModes, ncpcModes, tweeterModes]')
    parser.add_argument('shmim', type=str, help='Shared memory image [camsci1, camsci2, camlowfs, etc.]')
    parser.add_argument('core', type=float, help='Radius of the PSF core to measure')
    parser.add_argument('modes', type=str, help='Range of modes [x...y], comma-separated list of modes [x,y,z], or some combination of the two.')
    parser.add_argument('range', type=float, help='Range of values in microns over which to perform a grid search.')
    parser.add_argument('--nsteps', type=int, default=20, help='Number of points to sample in grid search [Default: 20]')
    parser.add_argument('--nrepeats', type=int, default=3, help='Number of sweeps [Default: 3]')
    parser.add_argument('--nclusterrepeats', type=int, default=1, help='Number of times to repeat a cluster of modes [Default: 1]')
    parser.add_argument('--nseqrepeat', type=int, default=1, help='Number of times to repeat the optimization of all modes [Default: 1]')
    parser.add_argument('--nimages', type=int, default=1, help='Number of images to collect from shmim [Default: 1]')
    parser.add_argument('--reset',  action='store_true', help='Ignore the current value of the mode and optimize about 0')

    args = parser.parse_args()

    # connect to INDI server
    client = indi.INDIClient('localhost', args.portINDI)
    client.start()

    # connect to shmim
    shmim = ImageStream(args.shmim)

    modes = parse_modes(args.modes)

    # run eye doctor
    eye_doctor_comprehensive(client, args.device, shmim, args.nimages, modes=modes, bounds=[-args.range/2., args.range/2], search_kind='grid',
                             search_dict={'nsteps' : args.nsteps, 'nrepeats' : args.nrepeats}, metric=get_image_coresum, metric_dict={'radius' : args.core},
                             ncluster=5, nrepeat=args.nclusterrepeats, nseqrepeat=args.nseqrepeat, randomize=True,
                             curr_prop='current_amps', targ_prop='target_amps', baseline=~args.reset)

def parse_modes(modestr):
    '''
    Parse command line inputs in the form
    '1...5,7,8,10...13' into
    a list like [1,2,3,4,5,7,8,10,11,12,13]
    '''
    comma_split = modestr.split(',')
    mode_list = []
    for c in comma_split:
        if '...' in c:
            m1, m2 = c.split('...')
            int1, int2 = int(m1), int(m2)
            if int1 > int2:
                clist = range(int1, int2-1, -1)
            else:
                clist = range(int1, int2+1)
            mode_list.extend(clist)
        else:
            mode_list.append(int(c))
    return mode_list

def main():
    '''
    Steps to getting to optimization:
    * Connect to client
    * Connect to shmim
        * If metric is 'pyramid', need to get pupil masks shmim too
    * Set up sequence(s)
        * allow something like modes='all',modes='low',modes='high', modes=2,3,4,7,10
        * need to fix defocus bug (try modes=1,2,3 -- it drops one of them)
    * Run optimization

    Required arguments (ugh):
    * dmModes (wooferModes, ncpcModes, tweeterModes)
    * camera shmim (camsci1, camsci2, lowfs??, etc)
    * pupil mask shmim(???)

    Optional arguments:
    * INDI port (default = 7624)
    * metric name (default = coresum?)
    * metric parameters (default = ummmmmmm.)
    * modes to optimize (default = 2...36)
    * sequence parameters (defaults=whatever)

    '''

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('portINDI', type=str, default=7624, help='INDI Port')
    parser.add_argument('device', type=str, help='INDI device with dm modes. [wooferModes, ncpcModes, tweeterModes]')
    parser.add_argument('shmim', type=str, help='Shared memory image [camsci1, camsci2, camlowfs, etc.]')
    parser.add_argument('metric', type=str, help='Optimization metric: "peak", "core", "ratio", or "pyramid"')
    parser.add_argument('modes', type=str, help='Range of modes [x...y] or comma-separated list of modes [x,y,z]')


    # parse args
    # allow something modes=1...10, modes=2,3,4,7,10, modes=2,10...36


    # connect to INDI server

    # connect to shmim

    # build sequence?

    # optimize!
    pass


if __name__ == '__main__':
    main()
