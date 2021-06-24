'''
Orphaned functions that I'm not quite willing to throw away yet.
'''

def fit_pupil_to_psf(camstream, nimages, sliceyx=None, padding=0, fwhm_guess=10):
    # measure an image
    image = np.mean(camstream.grab_many(nimages),axis=0).astype(float)
    if sliceyx is not None:
        image = image[sliceyx]
    meas_psf = image - np.median(image)
    meas_psf = shift_to_centroid(meas_psf)
    meas_psf = pad(meas_psf, padding)
    scale_factor, shifty, shiftx = fit_psf(meas_psf, get_magaox_pupil, fwhm_guess=fwhm_guess)[0]
    pupil, pupil_sampled = get_magaox_pupil(meas_psf.shape[0], grid_size=6.5*scale_factor)
    return pupil_sampled, scale_factor, shifty, shiftx  

def shift_to_centroid(im, shiftyx=None, order=3):
    if shiftyx is None:
        comyx = np.where(im == im.max())#com(im)
        cenyx = ((im.shape[0] - 1)/2., (im.shape[1] - 1)/2.)
        shiftyx = (cenyx[0]-comyx[0], cenyx[1]-comyx[1])
    median = np.median(im)
    return shift(im, shiftyx, order=1, cval=median)

def fit_psf(meas_psf, pupil_func, fwhm_guess=10.):
    shape = meas_psf.shape[0]
    pupil_guess = fwhm_guess / 2. 
    return leastsq(psf_err, [pupil_guess, 0, 0], args=(pupil_func, shape, meas_psf), epsfcn=0.01)

def psf_err(params, pupil_func, sampling , meas_psf):
    scale_factor, ceny, cenx = params
    _, pupil = pupil_func(sampling, grid_size=6.5*scale_factor)
    
    sim_psf = shift_to_centroid(fraunhofer_simulate_psf(pupil, 1, 0, np.zeros((sampling, sampling)), add_noise=False), (ceny, cenx))

    #print(rms(sim_psf-meas_psf, np.ones_like(sim_psf).astype(bool)))
    return (normalize_psf(sim_psf) - normalize_psf(meas_psf)).flatten()

def scale_and_noisify(psf, scale_factor, bg):
    psf_scaled = psf * scale_factor + bg
    poisson_noise = np.random.poisson(lam=np.sqrt(psf_scaled), size=psf_scaled.shape)
    return psf_scaled + poisson_noise

def normalize_psf(image):
    # normalize to unit energy
    return image / np.sum(image)