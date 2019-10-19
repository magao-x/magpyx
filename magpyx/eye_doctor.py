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
import ImageStreamIOWrap as shmio

from astropy.stats import sigma_clipped_stats
from astropy.io import fits

from scipy.optimize import leastsq, minimize_scalar, dual_annealing
from scipy.ndimage import center_of_mass

from scipy import stats
from scipy.special import jv

from skimage import draw


#-----purepyindi interaction-----

def get_value(client, device, prop, elem):
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
    return client[f'{device}.{prop}.{elem}']

def send_value(client, device, prop, elem, value):
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
    client[f'{device}.{prop}.{elem}'] = value
       
def get_zmode_vector(client, device, modes, prop='current_amps'):
    '''
    Get a vector of current zernike modes

    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
        prop : str
            Property name
        modes : list like
            List of zernike modes to query. Ex: range(3,7) or [1,3,6]

    Returns:
        amplitudes : list
            List of mode amplitudes
    '''
    amps = []
    for n in modes:
        elemname = '{:02}'.format(n)
        value = get_value(client, device, prop, elemname)
        amps.append(value)
    return amps

def send_zmode_amplitude(client, device, mode, amp, prop='target_amps'):
    '''
    Set a single mode amplitude

    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
        mode : str
            Zernike mode: '00', '01', '02', etc
        amp : float
            Mode coefficient in microns RMS.
            
    '''
    client[f'{device}.{prop}.{mode}'] = amp
    
def send_zmode_vector(client, device, modes, amps, prop='target_amps'):
    '''
    Set many mode amplitudes

    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
        modes : list like
            List of zernike modes. Ex: [1,2,3] or range(3,8)
        amps : list like
            List of mode coefficients
    
    modes and amps must be the same length.
    '''
    if len(modes) != len(amps):
        raise ValueError('modes and amps are different lengths!')
    
    for n, a in zip(modes, amps):
        mode = '{:02}'.format(n)
        #print(mode, a)
        #send_value(client, device, 'target_amps', mode, a)
        send_zmode_amplitude(client, device, mode, a, prop=prop)
        
def zero_dm(client, device, prop='current_amps'):
    '''
    Set all mode amplitudes to 0.

    Parameters:
        client : purepyindi.Client object
        device : str
            Device name
    '''
    nmodes = len(client.devices[device].properties[prop].elements)
    zeros = np.zeros(nmodes)
    send_zmode_vector(client, device, range(nmodes), zeros)

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

def get_image_peak(shmim, nimages):
    arrlist = grab_images(shmim, nimages)
    peaks = []
    for image in arrlist:
        im_bgsub = subtract_bg(image, stype=1)
        im_peak = find_peak(im_bgsub, stype=0)
        peaks.append(im_peak)
    avgpeak = np.mean(peaks)
    return avgpeak

def get_image_coresum(shmim, nimages, radius):
    # collect images
    arrlist = grab_images(shmim, nimages)
    
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
    return avgcore

def get_image_core_ring_ratio(shmim, nimages, radius1, radius2):
    # alternate approach: take ratios of images and then average
    
    # collect images
    arrlist = grab_images(shmim, nimages)
    
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
    print(np.sqrt(np.sum((measured-model)**2)), penalty/np.sqrt(np.sum(measured**2)))
    return np.sqrt(np.sum((measured-model)**2)) + penalty/np.sqrt(np.sum(measured**2))

def get_pupil_variance(shmim, nimages, pupil_mask):
    images = grab_images(shmim, nimages)
    return np.var(np.mean(images, axis=0)[pupil_mask])

#-----the eye doctor-----

def obj_func(amp, client, device, shmim, nmode, nimages, metric, metric_dict):
    '''
    Function to minimize when eye doctoring.

    TO DO: DOCUMENT ME.

    This is the function that basically does all the work.


    You should remove the sending and waiting for values from this func.
    This func should just take input image(s) and produce a scalar output.
    '''
    
    mode = '{:02}'.format(nmode)

    send_zmode_amplitude(client, device, mode, amp)
    sleep(0.02)

    # wait for confirmation
    updated = False
    while not updated:
        value = get_value(client, device, 'current_amps', mode)
        if np.isclose(value, amp): updated=True
        sleep(0.02)

    sleep(0.05)
    
    if metric == 'airy':
        arrlist = grab_images(shmim, nimages)
        avg = np.mean(arrlist, axis=0)
        
        wavelength = metric_dict['wavelength']
        fnum = metric_dict['fnum']
        pixscale = metric_dict['pixscale']
        cutout = metric_dict['cutout']
        penalty = metric_dict['penalty']

        params, avg_cutout = fit_airy_disk(avg, wavelength, fnum, pixscale, cutout=cutout)
        model = obscured_airy_disk(avg.max(), wavelength, fnum, pixscale,
                                   (params[0], params[1]), (cutout,cutout)) + params[2]
        return airy_metric(avg_cutout, model, penalty=metric_dict['penalty'])
    elif metric == 'peak': # assume peak height
        avgpeak = get_image_peak(shmim, nimages)
        return -avgpeak
    elif metric == 'core':
        radius = metric_dict['radius']
        avgcore = get_image_coresum(shmim, nimages, metric_dict['radius'])
        return -avgcore
    elif metric == 'ratio':
        radius1 = metric_dict['radius1']
        radius2 = metric_dict['radius2']
        ratio = get_image_core_ring_ratio(shmim, nimages, radius1, radius2)
        return ratio
    elif metric == 'pyramid':
        pupil_mask = metric_dict['pupil_mask']
        return get_pupil_variance(shmim, nimages, pupil_mask)
    elif metric == 'debug':
        return np.mean(grab_images(shmim, nimages),axis=0)
    else:
        raise ValueError('metric= {} but metric must be "peak", "airy", "core", "ratio", or "pyramid"!'.format(metric))
        
def optimize_strehl(client, device, shmim, nmode, nimages, bounds, metric, metric_dict={}, tol=1e-5):
    res = minimize_scalar(obj_func, bounds=bounds,
                          args=(client, device, shmim, nmode, nimages, metric, metric_dict),
                          method='bounded', options={'maxiter' : 100, 'xatol' : tol})
    return res['x']

def optimize_modes(client, device, shmim, nimages, modes, bounds, metric, metric_dict={}, curr_prop='current_amps', targ_prop='target_amps',
                   baseline=True, coreradius=10, kind='grid', search_dict={}):

    optimized_amps = []
    cores = [] # metric

    for i, n in enumerate(modes):

        mode = '{:02}'.format(n)
        
        if baseline: # center on current/previous values
            baseval = get_value(client, device, curr_prop, mode)
            curbounds = baseval + np.asarray(bounds)
        else:
            baseval = 0.
            curbounds = bounds
        
        initamp = baseval
        send_zmode_amplitude(client, device, mode, initamp, prop=targ_prop)
        sleep(.25)
        #initpeak = get_image_peak(shmim, nimages)
        #print('Initial Peak Height: {}'.format(initpeak))
        initcore = obj_func(initamp, client, device, shmim, n, nimages, metric, metric_dict) #get_image_coresum(shmim, nimages, coreradius) #10
        print('Mode {}: Initial Metric: {}'.format(mode, initcore))
        
        if kind == 'minimize':
            tol = search_dict.get(1e-5)
            best_amp = optimize_strehl(client, device, n, nimages, curbounds, metric, metric_dict=metric_dict, tol=tol)
        else:
            nsteps = search_dict.get('nsteps', 20)
            nrepeats = search_dict.get('nrepeats', 3)
            best_amp = grid_sweep(client, device, shmim, n, nimages, curbounds, nsteps, nrepeats, metric, metric_dict=metric_dict)
            #print(best_amp)
            if np.isnan(best_amp):
                best_amp = baseval
            
        print('Mode {}: Optimized from {} to {:.2} microns'.format(mode, initamp, best_amp))
        
        # set DM mode to this amplitude and check whether it really
        # improve the solutionbefore moving to next mode
        send_zmode_amplitude(client, device, mode, best_amp, prop=targ_prop)
        sleep(0.25)

        # take an image to get the peak height
        #avgpeak = get_image_peak(shmim, nimages)
        #print('Final Peak Height: {}.'.format(avgpeak))
        finalcore = obj_func(best_amp, client, device, shmim, n, nimages, metric, metric_dict) #get_image_coresum(shmim, nimages, coreradius) #10
        print('Mode {}: Final Metric: {}'.format(mode, finalcore))
        optimized_amps.append(best_amp)
        cores.append(finalcore)
        
        # reject solutions that don't seem to help
        #if finalcore > initcore:
        #    print('Final Core: {}. Accepting.'.format(finalcore))
        #    optimized_amps.append(best_amp)
        #else:
        #    print('Final Core: {}. Rejecting.'.format(finalcore))
        #    send_zmode_amplitude(client, device, mode, initamp)
        #    optimized_amps.append(initamp)

    return optimized_amps, cores

def grid_sweep(client, device, shmim, n, nimages, curbounds, nsteps, nrepeats, metric, metric_dict={}, debug=False):
    
    steps = np.linspace(curbounds[0], curbounds[1], num=nsteps, endpoint=True)
    
    if not debug:
        curves = np.zeros((nrepeats, nsteps))
        for i in range( nrepeats):
            for j, s in enumerate(steps):
                curves[i, j] = obj_func(s, client, device, shmim, n, nimages, metric, metric_dict)
            
    if debug:
        allimages = []
        for i in range(nrepeats):
            curimages = []
            for s in steps:
                images = obj_func(s, client, device, shmim, n, nimages, metric, metric_dict)
                curimages.append(images)
            allimages.append(curimages)
        return steps, np.asarray(allimages)
    
    # get the mean min
    if metric_dict['kind'] == 'mean':
        return np.mean(steps[np.argmin(metrics,axis=1)])
    elif metric_dict['kind'] == 'fit':
        # fit a quadratic
        # the problem here is that it could go bad
        
        # combine all sweeps into one dataset to fit
        c, b, a = np.polyfit( np.repeat(steps, nrepeats), curves.T.flatten(), deg=2)
        minima =  - b / (2 * c)
        mean = minima
        
        #minima = []
        #for curve in curves:
        #    c, b, a = np.polyfit(steps, curve, deg=2)
        #    minima.append( - b / (2 * c))
        #mean = np.mean(minima)
        if (mean < curbounds[0]) or (mean > curbounds[1]):
            print('Bad quadratic fit!')
            return np.nan
        else:
            return mean
    else:
        raise ValueError('kind must be "mean" or "fit"!')

def focus_sequence(client, device, params, do_zero_dm=False):
    '''
    In the future, you might want to be able to choose the modes
    in the sequence. The problem is that I want to treat focus
    separately from the others, so the logic starts to get messy.
    
    Might want option to decrease range as you go.
    Or unique range per mode.
    
    Might want option to decrease tolerance as you go.
    
    Maybe you should have a function that builds a dictionary
    of run parameters, and then have the focus_sequence parse
    that to run. That might help customization
    '''
    
    if do_zero_dm:
        zero_dm(client, device) # maybe don't hardcode the number of modes
       
    metric = []
    for p in params:
        optimized_amps, metrics = optimize_modes(*p)
        metric.extend(metrics)
    return metric

def build_sequence(client, device, shmim, nimages, coreradius, metric, metric_dict={}, modes=range(2,36),
                  ncluster=5, nrepeat=3, nseqrepeat=2, kind='grid', search_dict={}, randomize=True, baseline=True, bounds=[-5e-3, 5e-3]):
    
    modes = list(modes)

    if 2 in modes:
        nmodes = len(modes) - 1
    else:
        nmodes = len(modes)
        
    nfullgroups, mpartial = np.divmod(nmodes, ncluster) # remove focus

    mode_args = []

    for j in range(nseqrepeat):
        # always do focus first and by itself if requested
        if 2 in modes:
            for k in range(nrepeat):
                mode_args.append([2,])

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
        args.append((client, device, shmim, nimages, m, bounds, metric, metric_dict, baseline, coreradius, search_dict))
            
    return args

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
    parser.add_argument('device', type=str, help='INDI device with dm modes.')
    parser.add_argument('shmim', type=str, help='Shared memory image')
    parser.add_argument('metric', type=str, help='Optimization metric: "peak", "core", "ratio", or "pyramid"')
    parser.add_argument('modes', type=str, help='Range of modes [x...y] or comma-separated list of modes [x,y,z]')


    # parse args
    # allow something modes=1...10, modes=2,3,4,7,10, modes=2,10...36
    pass


if __name__ == '__main__':
    main()
