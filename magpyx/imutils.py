import numpy as np
try:
    import cupy as cp
except ImportError:
    print('Could not import cupy. You may lose functionality.')
    cp = None

import pyfftw
from scipy.optimize import leastsq
#from skimage.feature import register_translation
from skimage.registration import phase_cross_correlation as register_translation
from scipy.ndimage import shift

def rms(image,mask=None):
    return np.sqrt(np.mean(image[mask]**2))

def plane(indices, piston, tip, tilt):
    return piston + indices[1]*tip + indices[0]*tilt

def plane_error(params, indices, image, mask):
    delta = plane(indices,*params) - image
    return delta[mask].flatten()

def fit_plane(image, mask=None, indices=None):
    if indices is None:
        indices = np.indices(image.shape)
    return leastsq(plane_error, [0.,0.,0.], args=(indices, image, mask))[0]

def remove_plane(image, mask):
    pcoeffs = fit_plane(image, mask=mask)
    indices = np.indices(image.shape)
    return (image - plane(indices, *pcoeffs))

def rescale_and_pad(image, scale_factor, pad_to):
    rescaled = rescale(image, scale_factor, order=3)
    shape = rescaled.shape
    clip = lambda x: x if x > 0 else 0
    rough_pads = ( clip((pad_to-shape[0])/2 ), clip((pad_to-shape[0])/2) )
    padding = ((int(np.ceil(rough_pads[0])), int(np.floor(rough_pads[0]))),
               (int(np.ceil(rough_pads[1])), int(np.floor(rough_pads[1]))))
    return np.pad(rescaled, padding, 'constant', constant_values=0)

def register_images(imlist, sliceyx=None, upsample=1):
    if sliceyx is None:
        sliceyx = (slice(None), slice(None))
    # reference to first image in stack
    imref_full = imlist[0]
    imref = imref_full[sliceyx]   
    imreglist = []
    for im in imlist:
        # find shift
        shiftyx, error, diffphase = register_translation(imref, im[sliceyx], upsample_factor=upsample)
        # shift to reference
        imreglist.append( shift(im[sliceyx], shiftyx) ) 
    return imreglist

def get_gauss(sigma, shape, cenyx=None, xp=np):
    if cenyx is None:
        cenyx = xp.asarray([(shape[0])/2., (shape[1])/2.]) # no -1
    yy, xx = xp.indices(shape).astype(float) - cenyx[:,None,None]
    g = xp.exp(-0.5*(yy**2+xx**2)/sigma**2)
    return g / xp.sum(g)

def convolve_fft(in1, in2, force_real=False):
    out = ifft2_shiftnorm(fft2_shiftnorm(in1,norm=None)*fft2_shiftnorm(in2,norm=None),norm=None)
    if force_real:
        return out.real
    else:
        return out
    
def gauss_convolve(image, sigma, force_real=True):
    if cp is not None:
        xp = cp.get_array_module(image)
    else:
        xp = np
    g = get_gauss(sigma, image.shape, xp=xp)
    return convolve_fft(image, g, force_real=force_real)

def fft2_shiftnorm(image, axes=None, norm='ortho', shift=True):

    if axes is None:
        axes = (-2, -1)

    if isinstance(image, np.ndarray): # CPU or GPU
        xp = np
    else:
        xp = cp

    if shift:
        shiftfunc = xp.fft.fftshift
        ishiftfunc = xp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x

    if isinstance(image, np.ndarray):
        t = pyfftw.builders.fft2(ishiftfunc(image, axes=axes), axes=axes, threads=8, planner_effort='FFTW_ESTIMATE', norm=norm)
        return shiftfunc(t(),axes=axes)
    else:
        return shiftfunc(cp.fft.fft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm), axes=axes)
    
def ifft2_shiftnorm(image, axes=None, norm='ortho', shift=True):

    if axes is None:
        axes = (-2, -1)

    if isinstance(image, np.ndarray): # CPU or GPU
        xp = np
    else:
        xp = cp

    if shift:
        shiftfunc = xp.fft.fftshift
        ishiftfunc = xp.fft.ifftshift
    else:
        shiftfunc = ishiftfunc = lambda x, axes=None: x

    if isinstance(image, np.ndarray):
        t = pyfftw.builders.ifft2(ishiftfunc(image, axes=axes), axes=axes, threads=8, planner_effort='FFTW_ESTIMATE', norm=norm)
        return shiftfunc(t(), axes=axes)
    else:
        return shiftfunc(cp.fft.ifft2(ishiftfunc(image, axes=axes), axes=axes, norm=norm), axes=axes)

def rot_matrix(angle_rad, y=0, x=0):
    # rotation about the origin
    cos = np.cos(angle_rad)
    sin = np.sin(angle_rad)
    return np.asarray([[cos, sin, -y*cos-x*sin+y], [-sin, cos, y*sin-x*cos+x]]).T

def rotate(cy, cx, angle, ceny=0, cenx=0):
    return np.dot(rot_matrix(np.deg2rad(angle), y=ceny, x=cenx).T, np.asarray([cy, cx, np.ones(len(cy))]))

def shift_via_fourier(image, xk, yk, force_real=False,):
    xp = get_array_module(image)
    out =  ifft2_shiftnorm(fft2_shiftnorm(image, shift=False)*my_fourier_shift(xk, yk, image.shape, xp=xp), shift=False)
    if force_real:
        return out.real
    else:
        return out

def pad(image, padlen):
    val = np.median(image)
    return np.pad(image, padlen, mode='constant', constant_values=val)

