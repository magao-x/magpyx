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
        wait : bool
            Wait for property to register on INDI
        timeout: float
            Time out after X seconds if the INDI property never registers.
    
    Returns: element value
    '''
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
        wait : bool
            Wait for property to register on INDI
        timeout: float
            Time out after X seconds if the INDI property never registers.

    Returns: nothing
    
    '''
    if wait:
        client.wait_for_properties([f'{device}.{prop}',], timeout=timeout)
    client[f'{device}.{prop}.{elem}'] = value
        
def zero_dm(client, device):
    '''
    Set all mode amplitudes to 0.

    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
    '''
    client.wait_for_properties([f'{device}.current_amps', f'{device}.target_amps',], timeout=10)
    nmodes = len(client.devices[device].properties['current_amps'].elements)
    send_modes_and_wait(client, device, {m:0 for m in range(nmodes)})

def send_modes_and_wait(client, device, mode_targ_dict, tol=1e-3, wait_for_properties=True, timeout=10):
    '''
    Send target commands via purepyindi and return only after current becomes the target.

    Parameters:
        client: purepyindi.INDIClient
        device: str
            INDI device name. Ex: wooferModes
        mode_targ_dict : dict
            Dictionary of modes and their corresponding target values in microns Ex {0: 0.3, 3: 0.5, 10: -0.1}
        tol : float
            Return when abs(current - value) < tol
        wait_for_properties: bool
            Wait for INDI properties to register?
        timeout : float
            Wait for X seconds before failing
    '''
    status_dict = {}
    for mode, targ in mode_targ_dict.items():
        status_dict.update({
            f'{device}.current_amps.{mode:0>4}': {
                'value': targ,
                'test': lambda current, value, tolerance=tol: abs(current - value) < tolerance,
            #},
            #f'{device}.target_amps.{mode:0>4}': {
            #    'value': targ,
            }})
    return client.wait_for_state(status_dict, wait_for_properties=wait_for_properties, timeout=timeout)

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

    Parameters:
        val : float
            Mode amplitude
        client: purepyindi.INDIClient
        device : str
            INDI device name
        shmim : magpyx.utils.ImageStream
            camera image stream
        nmode : int
            Mode number
        nimages : int
            Number of images to collect from shmim
        metric : func
            Function that takes an image cube and returns a scalar
        metric_dict : dict
            keyword arguments to pass to metric func

    Returns:
        scalar output of metric, the value to minimize
    '''

    # move
    send_modes_and_wait(client, device, {f'{nmode:0>4}' : val})
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

def grid_sweep(client, device, shmim, n, nimages, curbounds, nsteps, nrepeats, metric, metric_dict, skind='fit', debug=False):
    '''
    Sweep over a range of mode amplitudes and compute some metric for the PSF quality at each point.
    Fit a quadratic and find the point that minimizes the metric.

    Parameters:
        client: purepyindi.INDIClient
        device : str
            INDI device name
        shmim : magpyx.utils.ImageStream
            camera image stream
        n : int
            Mode number
        nimages : int
            Number of images to collect from shmim
        curbounds : array-like
            2 element array-like with the lower and upper bounds of the search
        nsteps : int
            Number of steps to take. Sampled points follow np.linspace(curbounds[0], curbounds[1], num=nsteps)
        metric : func
            Function that takes an image cube and returns a scalar
        metric_dict : dict
            keyword arguments to pass to metric func
        skind : str
            If 'fit', find the minimum by fitting a quadratic. If 'mean', find the mean mode amplitude that gives a minimum.
            Default: 'fit'
        degug: bool
            Return sampled metric points rather than the fit point that minimizes the metric. 

    Returns:
        mode amplitude that minimizes the metric
    '''
    
    steps = np.linspace(curbounds[0], curbounds[1], num=nsteps, endpoint=True)
    
    curves = np.zeros((nrepeats, nsteps))
    for i in range( nrepeats):
        for j, s in enumerate(steps):
            # move
            send_modes_and_wait(client, device, {f'{n:0>4}' : s})
            #sleep(0.1)
            #measure
            images = shmim.grab_many(nimages)
            #metric
            curves[i, j] = metric(images, **metric_dict)

    # skip processing and just pass the metric values back
    if debug:
        return steps, curves

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
    Optimize a set of modes using a custom metric function evaluated from images from shmim as a function of 
    modal coefficient amplitudes.

    Can either perform a Brent or grid search.

    By default, this tries to minimize the negative core of the PSF
    as a proxy for the Strehl.

    Can pass in a custom metric function of the form:

    def my_metric(image_cube, kw_arg1=val1, kw_arg2=val2):
        return scalar_value_to_minimize

    where image_cube is a ZxYxX cube of images and keyword arguments are 
    specified by the metric_dict argument.

    Parameters:
        client: purepyindi.INDIClient
        device : str
            INDI device name
        shmim : magpyx.utils.ImageStream
            camera image stream
        nimages : int
            Number of images to collect from shmim
        modes : array-like
            Modes to optimize
        bounds : array-like
            2 element array-like with the lower and upper bounds of the search
        search_kind : str
            Either 'grid' or 'brent'. Default: grid (generally more robust)
        search_dict : dict
            Search parameters. For grid, accepts keywords 'kind' : 'mean' or 'fit',
            'nsteps' : int value, and 'nrepeats' : int value
        metric : func
            Function that takes an image cube and returns a scalar
        metric_dict : dict
            keyword arguments to pass to metric func
        curr_prop : str
            Current property. Default "current_amps". Will this ever change? I don't know.
        targ_prop : str
            Target property. Default "target_amps"
        baseline : bool
            Use the current mode amplitudes as the zero-point for the search? (centers search on 
            this point). Default: True
    '''

    # wait for devices to appear on INDI client
    client.wait_for_properties([f'{device}.{targ_prop}', f'{device}.{curr_prop}'])

    optimized_amps = []
    metric_vals = []
    # loop over modes
    for i, n in enumerate(modes):

        # baseline centers the search or sweep around the current value 
        if baseline:
            baseval = get_value(client, device, curr_prop, f'{n:0>4}')
            curbounds = np.asarray(bounds) + baseval
        else:
            baseval = 0.
            curbounds = bounds

        logger.info(f'Mode {n}: Scanning coefficients from {curbounds[0]} to {curbounds[1]} microns RMS')

        # measure
        meas_init = shmim.grab_many(nimages)
        # metric
        val_init = metric(meas_init, **metric_dict)

        # grid_sweep or optimize on metric
        # measurements outside of metric
        if search_kind == 'brent':
            tol = search_dict.get('tol', 1e-5)
            optval = optimize_strehl(client, device, n, nimages, curbounds, metric, metric_dict=metric_dict, tol=tol)
        elif search_kind == 'grid':
            nsteps = search_dict.get('nsteps', 20)
            nrepeats = search_dict.get('nrepeats', 3)
            optval = grid_sweep(client, device, shmim, n, nimages, curbounds,
                                nsteps, nrepeats, metric, metric_dict)
            if np.isnan(optval):
                optval = baseval
        else:
            raise ValueError('search_kind must be either "brent" or "grid".')

        # send to optimized value
        send_modes_and_wait(client, device, {f'{n:0>4}' : optval})
        #sleep(0.1)

        # measure
        meas_final = shmim.grab_many(nimages)
        # metric
        val_final = metric(meas_final, **metric_dict)

        logger.info(f'Mode {n}: Optimized metric {metric} from {val_init:.5} to {val_final:.5}')
        logger.info(f'Mode {n}: Optimized mode coefficient from {baseval:.5} to {optval:.5} microns')
        logger.info(f'-----------------------------------------------------------------------------')

def build_sequence(client, device, shmim, nimages, metric=get_image_coresum, metric_dict={}, search_kind='grid', search_dict={}, curr_prop=None, targ_prop=None,
                   modes=range(2,36), ncluster=5, nrepeat=3, nseqrepeat=2, randomize=True, baseline=True, bounds=[-5e-3, 5e-3]):
    '''
    Construct a sequence of argument tuples to send to eye_doctor.
    This is useful for building more complicated mode optimization
    procedures.

    Given a set of modes to optimize, this function splits them into smaller
    clusters, shuffles them about, and makes repeated passes in optimizing the
    modes.


    Parameters:
        client: purepyindi.INDIClient
        device : str
            INDI device name
        shmim : magpyx.utils.ImageStream
            camera image stream
        nimages : int
            Number of images to collect from shmim
        metric : func
            Function that takes an image cube and returns a scalar
        metric_dict : dict
            keyword arguments to pass to metric func
        search_kind : str
            Either 'grid' or 'brent'. Default: grid (generally more robust)
        search_dict : dict
            Search parameters. For grid, accepts keywords 'kind' : 'mean' or 'fit',
            'nsteps' : int value, and 'nrepeats' : int value
        curr_prop : str
            Current property. Default "current_amps". Will this ever change? I don't know.
        targ_prop : str
            Target property. Default "target_amps"
        modes : array-like
            Modes to optimize
        ncluster : int
            Number of elements in a cluster. Long lists of modes are split into clusters of
            this size.
        nrepeat : int
            Number of times to repeat the measurements in a given cluster.
        nseqrepeat: int
            Number of times to repeat the entire sequence.
        randomize: bool
            Shuffle the modes around within a cluster? Default: True.
        baseline : bool
            Use the current mode amplitudes as the zero-point for the search? (centers search on 
            this point). Default: True
        bounds : array-like
            2 element array-like with the lower and upper bounds of the search

    Returns: list of tuple arguments that can be passed to eye_doctor function
    '''
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
                             modes=range(2,36), ncluster=5, nrepeat=3, nseqrepeat=2, randomize=True, baseline=True, bounds=[-5e-3, 5e-3], focus_first=True):
    '''
    Perform a more involved / more global mode optimization
    procedure.

    Given a set of modes to optimize, this function splits them into smaller
    clusters, shuffles them about, and makes repeated passes in optimizing the
    modes.

    Parameters:
        client: purepyindi.INDIClient
        device : str
            INDI device name
        shmim : magpyx.utils.ImageStream
            camera image stream
        nimages : int
            Number of images to collect from shmim
        metric : func
            Function that takes an image cube and returns a scalar
        metric_dict : dict
            keyword arguments to pass to metric func
        search_kind : str
            Either 'grid' or 'brent'. Default: grid (generally more robust)
        search_dict : dict
            Search parameters. For grid, accepts keywords 'kind' : 'mean' or 'fit',
            'nsteps' : int value, and 'nrepeats' : int value
        curr_prop : str
            Current property. Default "current_amps". Will this ever change? I don't know.
        targ_prop : str
            Target property. Default "target_amps"
        modes : array-like
            Modes to optimize
        ncluster : int
            Number of elements in a cluster. Long lists of modes are split into clusters of
            this size.
        nrepeat : int
            Number of times to repeat the measurements in a given cluster.
        nseqrepeat: int
            Number of times to repeat the entire sequence.
        randomize: bool
            Shuffle the modes around within a cluster? Default: True.
        baseline : bool
            Use the current mode amplitudes as the zero-point for the search? (centers search on 
            this point). Default: True
        bounds : array-like
            2 element array-like with the lower and upper bounds of the search
        focus_first : bool
            Start with focus? Default: True.

    Returns: list of tuple arguments that can be passed to eye_doctor function
    '''

    # reject modes that aren't allowed
    # for now, hardcode to first 36 zernikes
    allowed_modes = [m for m in modes if m in range(36)]
    if len(allowed_modes) < len(modes):
        logger.warning('Not correcting modes > 35.')

    if not baseline:
        logger.info('Not using the current baseline: resetting all mode coefficients to 0.')
        zero_dm(client, device)

    logger.info(f'Optimizing modes: {allowed_modes}')

    # build sequence
    args_seq = build_sequence(client, device, shmim, nimages, metric=metric, metric_dict=metric_dict, search_kind=search_kind,
                              search_dict=search_dict, curr_prop=curr_prop, targ_prop=targ_prop, modes=allowed_modes, ncluster=ncluster,
                              nrepeat=nrepeat, nseqrepeat=nseqrepeat, randomize=randomize, baseline=baseline, bounds=bounds)

    # if focus is requested, do it first (it may get repeated later but oh well)
    if focus_first:
        logger.info('Optimizing focus first!')
        eye_doctor(client, device, shmim, nimages, [2,], bounds, search_kind=search_kind, search_dict=search_dict, metric=metric,
                   metric_dict=metric_dict, curr_prop=curr_prop, targ_prop=targ_prop, baseline=baseline)

    # do the eye doctor
    for args in args_seq:
        eye_doctor(*args)

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
               curr_prop='current_amps', targ_prop='target_amps', baseline=not args.reset)

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
    parser.add_argument('--ignorefocus', action='store_true',
        help='By default, if more than one mode is being optimized, the eye doctor starts with focus. Use this option to turn this off.')

    args = parser.parse_args()

    # connect to INDI server
    client = indi.INDIClient('localhost', args.portINDI)
    client.start()

    # connect to shmim
    shmim = ImageStream(args.shmim)

    modes = parse_modes(args.modes)

    if len(modes) == 1 or args.ignorefocus:
        focus_first = False
    else:
        focus_first = True

    # run eye doctor
    eye_doctor_comprehensive(client, args.device, shmim, args.nimages, modes=modes, bounds=[-args.range/2., args.range/2], search_kind='grid',
                             search_dict={'nsteps' : args.nsteps, 'nrepeats' : args.nrepeats}, metric=get_image_coresum, metric_dict={'radius' : args.core},
                             ncluster=5, nrepeat=args.nclusterrepeats, nseqrepeat=args.nseqrepeat, randomize=True,
                             curr_prop='current_amps', targ_prop='target_amps', baseline=not args.reset, focus_first=focus_first)

def write_new_flat(dm, filename=None, update_symlink=False, overwrite=False):
    '''
    Write out a new flat to FITS after eye doctoring.

    Parameters:
        dm : str
            Name of DM. If not "woofer", "ncpc", or "tweeter", this is
            taken to be the name of a shared memory image
        filename : str
            FITS file to write out. If not provided, defaults to 
            /opt/MagAOX/calib/dm/<dm_name/flats/flat_optimized_<dm_name>_<date>.fits
        update_symlink:
            Update the symlink at /opt/MagAOX/calib/dm/<dm_name/flat.fits? Default: False
    '''
    from astropy.io import fits
    from datetime import datetime
    import os

    if dm.upper() == 'WOOFER':
        outpath = '/opt/MagAOX/calib/dm/alpao_bax150/'
        dm_name = 'ALPAOBAX150'
        shm_name = 'dm00disp'
    elif dm.upper() == 'NCPC':
        outpath = '/opt/MagAOX/calib/dm/alpao_bax260/'
        dm_name = 'ALPAOBAX260'
        shm_name = 'dm02disp'
    elif dm.upper() == 'TWEETER':
        outpath = '/opt/MagAOX/calib/dm/bmc_2k/'
        dm_name = 'BMC2K'
        shm_name = 'dm01disp'
    else:
        logger.warning('Unknown DM provided. Interpreting as a shared memory image name.')
        shm_name = dm
        dm_name = dm
        outpath = '.'

    if filename is None:
        date = datetime.now().strftime("%Y%m%d-%H%M%S")
        outname = os.path.join(outpath, 'flats', f'flat_optimized_{dm_name}_{date}.fits')
    else:
        outname = filename

    shmim = ImageStream(shm_name)
    flat = shmim.grab_latest()

    logger.info(f'Writing image at {shm_name} to {outname}.')
    fits.writeto(outname, flat, overwrite=overwrite)

    if update_symlink:
        sym_path = os.path.join(outpath, 'flat.fits')
        if os.path.islink(sym_path):
            os.remove(sym_path)
        os.symlink(outname, sym_path)
        logger.info(f'Symlinked image at {outname} to {sym_path}.')

def console_write_new_flat():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dm', type=str, help='DM. One of ["woofer", "ncpc", "tweeter"]')
    parser.add_argument('--filename', default=None, type=str,
                        help='Path of file to write out. Default: /opt/MagAOX/calib/dm/<dm_name>/flats/flat_optimized_<dm name>_<date/time>.fits')
    parser.add_argument('--symlink', action='store_true',
                        help='Symlink flat to /opt/MagAOX/calib/dm/<dm_name>/flats/flat.fits or at the custom path provided with the "--filename" flag.')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite an existing file? [Default: False]')
    args = parser.parse_args()
    write_new_flat(args.dm, filename=args.filename, update_symlink=args.symlink, overwrite=args.overwrite)

def console_zero_all_modes():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dmDevice', type=str, help='DM. One of ["wooferModes", "ncpcModes", "tweeterModes"]')
    parser.add_argument('--portINDI', type=int, default=7624, help='INDI Port on which [dm]Modes can be found. Default: 7624.')
    args = parser.parse_args()

    # start a client
    client = indi.INDIClient('localhost', args.portINDI)
    client.start()

    # zero the DM
    zero_dm(client, args.dmDevice)


def console_update_flat():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dm', type=str, help='DM. One of ["woofer", "ncpc", "tweeter"]')
    parser.add_argument('--portINDI', type=int, default=7624, help='INDI Port on which [dm]Modes can be found. Default: 7624.')
    args = parser.parse_args()

    dm = args.dm
    port = args.portINDI

    # write out new flat and symlink
    write_new_flat(dm, update_symlink=True)

    # zero modes
    client = indi.INDIClient('localhost', port)
    client.start()
    if dm.upper() == 'WOOFER':
        device = 'wooferModes'
        dmdevice = 'dmwoofer'
    elif dm.upper() == 'NCPC':
        device = 'ncpcModes'
        dmdevice = 'dmncpc'
    elif dm.upper() == 'TWEETER':
        device = 'tweeterModes'
        dmdevice = 'dmtweeter'
    else:
        raise ValueErorr('Unknown DM provided. Must be one of "woofer", "tweeter", or "ncpc".')

    logger.info(f"Cleared all modes on {device}.")
    zero_dm(client, device)

    # toggle reload flat
    status_dict = {f'{dmdevice}.flat.target': { 'value': 'flat.fits'}}
    client.wait_for_state(status_dict, wait_for_properties=True, timeout=10)
    sleep(3) # I don't know why I need this.
    logger.info(f"Reloaded flat on {dmdevice}.")

if __name__ == '__main__':
    pass
