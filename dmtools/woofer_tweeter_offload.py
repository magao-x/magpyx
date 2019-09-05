import os

import numpy as np
from astropy.io import fits
from scipy import linalg

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
    
    sinv = np.diag(1./s) # compute the inverse (this could create NaNs)
    sinv[reject] = 0. #remove elements that don't meet the threshold
    
    # just to be safe, remove any NaNs or infs
    sinv[np.isnan(sinv)] = 0.
    sinv[np.isinf(sinv)] = 0.
    
    # compute the pseudo-inverse: Vh.T s^-1 U_dagger (hermitian conjugate)
    return np.dot(Vh.T, np.dot(sinv, U.T.conj())), threshold, U, s, Vh

def plot_singular_values(sing_vals, threshold=None):
    
    fig, ax = plt.subplots(1, 1, figsize=(5,4))
    ax.semilogy(range(1, n+1), sing_vals,label='Sing. Values')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if threshold is not None:
        ax.hlines(threshold, xlim[0], xlim[1], linestyles='--', label='Threshold={:.2e}'.format(threshold))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('N (woofer modes)')
    ax.set_ylabel('Singular Value')
    ax.legend()
    
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
    parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite existing FITS file? Default=False')
    parser.add_argument('--inverse', type=bool, default=False, help='Compute the woofer -> tweeter offload matrix instead.')

    args = parser.parse_args()

    # read in response matrix cubes
    with fits.open(args.zrespM_woofer) as f:
        zrespM_woofer = f[0].data

    with fits.open(args.zrespM_tweeter) as f:
        zrespM_tweeter = f[0].data

    # compute the offload matrix
    if not args.inverse:
        wt_matrix = compute_tweeter_woofer_matrix(zrespM_woofer.reshape(zrespM_woofer.shape[0], -1).T,
                                                  zrespM_tweeter.reshape(zrespM_tweeter.shape[0], -1).T,
                                                  abs_threshold=args.abs_threshold,
                                                  rel_threshold=args.rel_threshold,
                                                  n_threshold=args.n_threshold)[0].T
    else: # compute the inverse
        wt_matrix = compute_woofer_tweeter_matrix(zrespM_woofer.reshape(zrespM_woofer.shape[0], -1).T,
                                                  zrespM_tweeter.reshape(zrespM_tweeter.shape[0], -1).T,
                                                  abs_threshold=args.abs_threshold,
                                                  rel_threshold=args.rel_threshold,
                                                  n_threshold=args.n_threshold)[0].T

    # write to file
    print('Writing {}x{} matrix to {}'.format(wt_matrix.shape[0], wt_matrix.shape[1], os.path.abspath(args.outname)))
    fits.writeto(args.outname, wt_matrix, overwrite=args.overwrite)


if __name__ == '__main__':
    main()