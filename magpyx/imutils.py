
import numpy as np
from scipy.optimize import leastsq
from skimage.feature import register_translation
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

def register_images(imlist, sliceyx=None):
    # reference to first image in stack
    imref_full = imlist[0]
    imref = imref_full[sliceyx]   
    imreglist = []
    for im in imlist:
        # find shift
        shiftyx, error, diffphase = register_translation(imref, im[sliceyx], 10)
        # shift to reference
        imreglist.append( shift(im[sliceyx], shiftyx) ) 
    return imreglist