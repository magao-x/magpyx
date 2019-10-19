import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('offload')

from .project_zernikes import projected_basis

def pseudoinverse_svd(matrix, abs_threshold=None, rel_threshold=None, n_threshold=None):
    '''
    Compute the pseudo-inverse of a matrix via an SVD and some threshold.
    
    Only one type of threshold should be specified.
    
    Parameters:
        matrix: nd array
            matrix to invert
        abs_threshold: float
            Absolute value of sing vals to threshold
        rel_threshold: float
            Threshold sing vals < rel_threshold * max(sing vals)
        n_threshold : int
            Threshold beyond the first n_threshold singular values
        
    Returns:
        pseudo-inverse : nd array
            pseduo-inverse of input matrix
        threshold : float
            The absolute threshold computed
        U, s, Vh: nd arrays
            SVD of the input matrix
    '''
    from scipy import linalg

    
    if np.count_nonzero([abs_threshold is not None,
                         rel_threshold is not None,
                         n_threshold is not None]) > 1:
        raise ValueError('You must specify only one of [abs_threshold, rel_threshold, n_threshold]!')
        
    # take the SVD
    U, s, Vh = np.linalg.svd(matrix, full_matrices=False)
        
    #threshold
    if abs_threshold is not None:
        threshold = abs_threshold
        reject = s <= threshold
    elif rel_threshold is not None:
        threshold = s.max() * rel_threshold
        reject = s <= threshold
    elif n_threshold is not None:
        reject = np.arange(len(s)) > n_threshold
        threshold = s[n_threshold]
    else:
        threshold = 1e-16
        reject = s <= threshold
    
    sinv = np.diag(1./s).copy() # compute the inverse (this could create NaNs)
    sinv[reject] = 0. #remove elements that don't meet the threshold
    
    # just to be safe, remove any NaNs or infs
    sinv[np.isnan(sinv)] = 0.
    sinv[np.isinf(sinv)] = 0.
    
    # compute the pseudo-inverse: Vh.T s^-1 U_dagger (hermitian conjugate)
    return np.dot(Vh.T, np.dot(sinv, U.T.conj())), threshold, U, s, Vh

def plot_singular_values(sing_vals, threshold=None, semilogy=True, dm='woofer'):
    
    fig, ax = plt.subplots(1, 1, figsize=(6,5))
    if semilogy:
        ax.semilogy(range(1, len(sing_vals)+1), sing_vals,label='Sing. Values')
    else:
        ax.plot(range(1, len(sing_vals)+1), sing_vals,label='Sing. Values')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if threshold is not None:
        ax.hlines(threshold, xlim[0], xlim[1], linestyles='--', label='Threshold={:.2e}'.format(threshold))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(f'N ({dm} modes)')
    ax.set_ylabel('Singular Value')
    ax.legend()
    fig.tight_layout()
    
    return fig, ax

def compute_tweeter_woofer_matrix(woofer_respM, tweeter_respM, **kwargs):
    '''
    Tweeter -> woofer offload
    '''
    woofer_inv, threshold, U, s, Vh = pseudoinverse_svd(woofer_respM, **kwargs)
    #woofer_inv =  pinv2(woofer_respM)#, rcond=5e-3)
    return np.dot(woofer_inv, tweeter_respM), s, threshold

def compute_woofer_tweeter_matrix(woofer_respM, tweeter_respM, **kwargs):
    '''
    Woofer -> tweeter offload
    '''
    tweeter_inv, threshold, U, s, Vh = pseudoinverse_svd(tweeter_respM, **kwargs)
    return np.dot(tweeter_inv, woofer_respM), s, threshold

def mode_crosscheck(wt_matrix, dm1_mask, dm2_mask, dm1='tweeter', dm2='woofer'):
    '''
    After computing the offload matrix, feed it modes
    of known RMS and check that the offloaded-values
    have a similar RMS.
    '''

    if isinstance(dm1_mask, str):
        with fits.open(dm1_mask) as f:
            dm1_mask = f[0].data.astype(bool)
    if isinstance(dm2_mask, str):
        with fits.open(dm2_mask) as f:
            dm2_mask = f[0].data.astype(bool)

    # generate dm1 modes (default=tweeter)
    nmodes = 36
    tmodes = projected_basis(nmodes, 0., 1.0, dm1_mask)

    # project onto dm2 (default=woofer)
    wmodes = []
    for t in tmodes:
        wmodes.append(np.dot(wt_matrix, t.flatten()).reshape(dm2_mask.shape))

    # get rms values
    rmsvals_dm1 = [np.sqrt(np.mean(t[dm1_mask]**2)) for t in tmodes]
    rmsvals_dm2 = [np.sqrt(np.mean(w[dm2_mask]**2)) for w in wmodes]

    # report (downgrade this to info and fix notebook log outputs)
    logger.info(f'Mode #: {dm1} RMS ---> {dm2} RMS (microns)')
    for n in range(nmodes):
        logger.info('Mode {}: {:.2f} ---> {:.2f}'.format(n, rmsvals_dm1[n], rmsvals_dm2[n]))


def get_offload_matrix(zrespM_woofer, zrespM_tweeter, outname=None, abs_threshold=None, rel_threshold=None, n_threshold=None, crosscheck=False, display=False, overwrite=False, inverse=False):

    # read in response matrix cubes
    with fits.open(zrespM_woofer) as f:
        zrespM_woofer = f[0].data

    with fits.open(zrespM_tweeter) as f:
        zrespM_tweeter = f[0].data

    # actuator masks
    tweeter_mask = '/opt/MagAOX/calib/dm/bmc_2k/bmc_2k_actuator_mask.fits'
    woofer_mask = '/opt/MagAOX/calib/dm/alpao_bax150/bax150_actuator_mask.fits'

    # compute the offload matrix
    if not inverse:
        wt_matrix, sing_vals, threshold = compute_tweeter_woofer_matrix(zrespM_woofer.reshape(zrespM_woofer.shape[0], -1).T,
                                                  zrespM_tweeter.reshape(zrespM_tweeter.shape[0], -1).T,
                                                  abs_threshold=abs_threshold,
                                                  rel_threshold=rel_threshold,
                                                  n_threshold=n_threshold)
        dm1 = 'tweeter'
        dm2 = 'woofer'
        dm1_mask = tweeter_mask
        dm2_mask = woofer_mask
    else: # compute the inverse
        wt_matrix, sing_vals, threshold = compute_woofer_tweeter_matrix(zrespM_woofer.reshape(zrespM_woofer.shape[0], -1).T,
                                                  zrespM_tweeter.reshape(zrespM_tweeter.shape[0], -1).T,
                                                  abs_threshold=abs_threshold,
                                                  rel_threshold=rel_threshold,
                                                  n_threshold=n_threshold)
        dm1 = 'woofer'
        dm2 = 'tweeter'
        dm1_mask = woofer_mask
        dm2_mask = tweeter_mask

    nmodes = np.count_nonzero(sing_vals > threshold)
    logger.info(f'Thresholding after first {nmodes} singular values.')

    if crosscheck:
        mode_crosscheck(wt_matrix, dm1_mask, dm2_mask, dm1, dm2)
    if display:
        plot_singular_values(sing_vals, threshold=threshold, semilogy=True, dm=dm2)

    # write to file
    if outname is not None:
        logger.info('Writing {}x{} matrix to {}'.format(wt_matrix.shape[0], wt_matrix.shape[1], os.path.abspath(outname)))
        fits.writeto(outname, wt_matrix.T, overwrite=overwrite)

    return wt_matrix.T, sing_vals, threshold

def main():

    # parse command line arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('zrespM_woofer', type=str, help='Path to zrespM FITS file for the woofer')
    parser.add_argument('zrespM_tweeter', type=str, help='Path to zrespM FITS file for the tweeter')
    parser.add_argument('outname', type=str, help='Name/path of FITS file to write the tweeter-woofer offload matrix to.')
    parser.add_argument('--abs_threshold', type=float, default=None, help='Absolute threshold for singular values of woofer pseudoinverse. If no thresholds are given, the default is 1e-16.')
    parser.add_argument('--rel_threshold', type=float, default=None, help='Threshold as a fraction of the largest singular value')
    parser.add_argument('--n_threshold', type=int, default=None, help='Number of singular values to keep for woofer pseudoinverse.')
    parser.add_argument('--crosscheck', action='store_true', help='Generate modes, project onto offloaded basis, and report RMS values.')
    parser.add_argument('--display', action='store_true', help='Plot singular values.')
    parser.add_argument('--overwrite', action='store_true', default=False, help='Overwrite existing FITS file? Default=False')
    parser.add_argument('--inverse', action='store_true', default=False, help='Compute the woofer -> tweeter offload matrix instead.')

    args = parser.parse_args()

    get_offload_matrix(args.zrespM_woofer, args.zrespM_tweeter, outname=args.outname, abs_threshold=args.abs_threshold,
                       rel_threshold=args.rel_threshold, n_threshold=args.n_threshold, crosscheck=args.crosscheck,
                       display=args.display, overwrite=args.overwrite, inverse=args.inverse)

    

if __name__ == '__main__':
    main()