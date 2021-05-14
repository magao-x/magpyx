'''
All closed loop / DM control related functionality goes here.

To do:
* create closed loop integrator func
* create hadamard interaction measurement func (generalize what you have for FDPR and then refactor that to use this func)
'''
import numpy as np
from scipy.linalg import svd
from skimage.filters import threshold_otsu

import .dmutils

def collect_hadamard_interaction_matrix(dmstream, wfsfunc, paramdict={}):
    '''
    Generalized function for (slow) collection of a +/- hadamard interaction matrix.
    '''
    raise NotImplementedError('woops')

def closed_loop(dmstream, ctrlmat, wfsfunc, niter=10, gain=0.5, leak=0., paramdict={}):
    '''
    Generalized function for (slow) leaky integrator closed-loop operations with shmims.

    This function will handle calculating commands and writing to the DM,
    with the expectation that the user provides the wfsfunc that takes paramdict keyword
    arguments and passes back a WFS measurement compatible with the provided control
    matrix.
    '''
    raise NotImplementedError('woops')

def get_control_matrix_from_hadamard_measurements(hmeas, hmodes, hval, dm_map, dm_mask, wfsthresh=0.5, dmthresh=0.5, ninterp=2, nmodes=None):
    '''
    Given a measurement cube of WFS measurements of +/- hadamard modes with amplitude hval,
    return the reconstructed IF matrix, DMmodes, WFSmodes, DM and WFS maps/masks, and the control/reconstructor matrix
    '''
    # construct IF cube
    ifcube = get_ifcube_from_hmeas(hmeas, hmodes, hval)

    # get SVD
    wfsmodes, singvals, dmmodes = get_svd_from_ifcube(ifcube)

    # get DM and WFS maps and masks
    wfs_ctrl_map, wfs_ctrl_mask = get_wfs_ctrl_map_mask(ifcube, threshold=wfsthresh)
    dm_ctrl_map, dm_ctrl_mask = get_wfs_ctrl_map_mask(ifcube, dm_map, dm_mask, threshold=dmthresh)

    # interpolate DM modes
    dmmodes_interp = interpolate_dm_modes(dmmodes, dm_ctrl_mask, dm_map, dm_mask, n=ninterp)

    # get control matrix
    C = get_control_matrix(dmmodes_interp, wfsmodes, singvals, nmodes=nmodes)

    return {
        'ifcube' : ifcube,
        'wfs_ctrl_map' : wfs_ctrl_map,
        'wfs_ctrl_mask' : wfs_ctrl_mask,
        'dm_ctrl_map' : dm_ctrl_map,
        'dm_ctrl_mask' : dm_ctrl_mask,
        'wfsmodes' : wfsmodes,
        'dmmodes' : dmmodes_interp,
        'singvals' : singvals,
        'C' : C
    }

def get_control_matrix(dmmodes, wfsmodes, singvals, nmodes=None):
    '''
    Compute the control matrix from a pseudo-inverse via regularized SVD components
    '''
    nact = dmmodes.shape[0]
    nwfs = wfsmodes.shape[0]

    if nmodes is None:
        nmodes = nact

    # construct regularized S^{-1}
    Sinv = np.zeros((nact, nwfs))
    sinv = 1./singvals
    sinv[nmodes:] *= 0
    np.fill_diagonal(Sinv, sinv)

    # pseudo-inverse calc
    C = dmmodes.T.dot(Sinv.dot(wfsmodes.T))

    return C

def get_svd_from_ifcube(ifcube):
    '''
    Compute the SVD of the IF matrix F s.t. WFSmodes @ S @ DMmodes = F
    '''
    nact = ifcube.shape[0]
    wfsmodes, s, dmmodes = svd(ifcube.reshape((nact,-1)).swapaxes(0,1))

def get_ifcube_from_hmeas(hmeas, hmodes, hval):
    '''
    Reconstruct the influence function cube from the measured WFS
    response to +/- hadamard modes
    '''
    shape = hmodes.shape # better be a square matrix
    shape_wfs = hmeas.shape[-2:]
    if shape[0] != shape[1]:
        raise ValueError(f'hmodes must be a square matrix, but got shape {shape}!')

    # take the difference of the +/- measurements and normalize by the commanded amplitude hval
    hcube_norm = (hmeas[:shape[0]] - hmeas[:shape[1]]) / 2. / hval

    return np.dot(hmodes, hcube_norm.reshape(shape[0],-1)).reshape(shape_wfs)

def get_dm_ctrl_map_mask(ifcube, dm_map, dm_mask, threshold=0.5):
    '''
    Given a Nact x Nwfs x Nwfs cube of influence functions, return a dm control map and mask

    Parameters
        ifcube : ndarray
            Nact x Nwfs x Nwfs cube of influence functions
        dm_map : ndarray
            Nact x Nact map of actuator positions (from the /calib/dm/<dm> directory)
        dm_mask : ndarray
            Nact x Nact mask of active actuators (from the /calib/dm/<dm> directory)
        threshold : float
            Actuators w/ RMS response < (threshold * otsu_threshold) will be considered
            inactive and couple-controlled

    Returns
        dm_ctrl_map : ndarray
            Nact x Nact array of RMS IF responses
        dm_ctrl_mask : ndarray
            Nact x Nact binary array of actively-controlled/illuminated actuators
    '''
    rms_dm = np.sqrt(np.mean(ifcube**2,axis=(-2,-1)))
    dm_ctrl_map = dmutils.map_vector_to_square(rms_dm, dm_map, dm_mask)
    thresh_dm = threshold_otsu(dm_ctrl_map)
    dm_ctrl_mask = dm_ctrl_map > (thresh_dm*threshold)

    return dm_ctrl_map, dm_ctrl_mask

def get_wfs_ctrl_map_mask(ifcube, threshold=0.5):
    '''
    Given a Nact x Nwfs x Nwfs cube of influence functions, return a dm control map and mask

    Parameters
        ifcube : ndarray
            Nact x Nwfs x Nwfs cube of influence functions
        dm_map : ndarray
            Nact x Nact map of actuator positions (from the /calib/dm/<dm> directory)
        dm_mask : ndarray
            Nact x Nact mask of active actuators (from the /calib/dm/<dm> directory)
        threshold : float
            Actuators w/ RMS response < (threshold * otsu_threshold) will be considered
            inactive and couple-controlled

    Returns
        wfs_ctrl_map : ndarray
            Nwfs x Nwfs array of RMS WFS response (pixel by pixel)
        wfs_ctrl_mask : ndarray
            Nwfs x Nwfs binary array of WFS pixels that respond to DM commands
    '''
    shape = ifcube.shape
    wfs_ctrl_map = np.sqrt(np.mean(ifcube**2,axis=0)).reshape((shape[1],shape[2]))
    thresh_wfs = threshold_otsu(wfs_ctrl_map)
    wfs_ctrl_mask = wfs_ctrl_map > (thresh_wfs*threshold)

    return wfs_ctrl_map, wfs_ctrl_mask

def interpolate_dm_modes(dm_mode_matrix, active_mask, dm_map, dm_mask, n=1):
    
    # get the interpolation mapping
    couple_matrix = get_coupling_matrix(active_mask, dm_map, dm_mask, n=n)
    n = couple_matrix.shape[0]
    
    # a matrix multiply does the interpolation
    interpolated_modes = np.dot(dm_mode_matrix, np.eye(n) + couple_matrix)
    
    return interpolated_modes

def get_coupling_matrix(active_mask, dm_map, dm_mask, n=1):
    '''
    This is so convoluted. Good luck.
    '''

    # define inactive map
    inactive_mask = ~active_mask & dm_mask
    nact = np.sum(dm_mask)

    # get indices for distance calc later
    shape = dm_map.shape
    indices = np.indices(shape)

    # matrix
    couple_matrix = np.zeros((nact, nact)) 

    # for each inactive actuator, figure out the n closest active actuators
    for y, x in np.squeeze(np.dstack(np.where(inactive_mask))):
        idy = indices[0] - y
        idx = indices[1] - x
        d = np.sqrt(idy**2 + idx**2)
        #np.argsort(np.ma.masked_where(active_mask, d), axis=None)
        dsort = np.unravel_index(np.argsort(np.ma.masked_where(~active_mask, d), axis=None), d.shape)
        neighbor_map = np.zeros_like(d)
        neighbor_map[dsort[0][:n], dsort[1][:n]] = 1/n

        tmp = np.zeros(shape)
        tmp[y,x] = 1
        act_idx = np.where(dmutils.map_square_to_vector(tmp, dm_map, dm_mask))[0][0]
        couple_matrix[act_idx] = dmutils.map_square_to_vector(neighbor_map, dm_map, dm_mask)
    
    return couple_matrix.T # not sure why I need this transpose, but, uhh