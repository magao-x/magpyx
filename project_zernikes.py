#!/usr/bin/env python

import argparse
from datetime import datetime
import numpy as np
from astropy.io import fits
from poppy.zernike import R, noll_indices

def zernike(n, m, npix=100, aperture=None, rho=None, theta=None, outside=np.nan,
            noll_normalize=True, **kwargs):
    """Return the Zernike polynomial Z[m,n] for a given pupil.

    poppy.zernike.zernike sets values for rho > 1 to 0, so this is a slightly
    modified duplicate of that function that leaves those values alone, as a
    kludgy way of extrapolating DM commands outside the beam footprint.
    """

    if aperture is None:
        aperture = np.ones((npix, npix))

    if m == 0:
        if n == 0:
            zernike_result = aperture
        else:
            norm_coeff = np.sqrt(n + 1) if noll_normalize else 1
            zernike_result = norm_coeff * R(n, m, rho) * aperture
    elif m > 0:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.cos(np.abs(m) * theta) * aperture
    else:
        norm_coeff = np.sqrt(2) * np.sqrt(n + 1) if noll_normalize else 1
        zernike_result = norm_coeff * R(n, m, rho) * np.sin(np.abs(m) * theta) * aperture

    return zernike_result

def zernike_basis(nterms, nact, angle, fill_fraction):
    '''
    Return a set of Zernike modes stretched such that
    they form ordinary Zernike shapes when deprojected.

    Values outside the beam footprint are extrapolated
    by evaluting the terms for rho > 1.

    Zernike modes are ordered by the Noll convention
    and normalized such that they have a unit RMS in
    the deprojected beam footprint.

    Parameters:
        nterms : int
            Number of terms in the basis. Always skips piston.
        nact : int
            Number of actuators across the pupil. For the 2K,
            this is 50. For the ALPAOs, this is 11.
        angle : float
            Incidence angle of the beam on the DM, given in
            degrees. Angles are always assumed to be in the x-z
            plane.
        fill_fraction : float
            Fraction of the DM in the beam footprint (defined
            along the x axis) [=0.96 for the 2K]

    Returns: list of nterms modes of shape nact x act 
    '''

    # generate (n, m) indices for zernike modes (skip piston)
    nmlist = [noll_indices(j) for j in range(2,nterms+2)]

    # generate the radial and theta arrays that account for geometrical
    # projection
    cos = np.cos(np.deg2rad(angle))
    x = np.linspace(-(nact-1)/nact, (nact-1)/nact, num=nact, endpoint=True) / fill_fraction
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + (yy/cos)**2)
    theta = np.arctan2(yy/cos, xx)

    # make the modes
    zbasis = [zernike(n, m, npix=nact, outside=0., rho=rho, theta=theta) for (n, m) in nmlist]

    return zbasis

def write_fits(zbasis, outname, nact, angle, fill_fraction, overwrite=False):
    '''Write zernike basis out to FITS file'''

    hdu = fits.PrimaryHDU(np.asarray(zbasis, dtype=np.float32))
    hdu.header.update({
        'nact' : nact,
        'angle' : angle,
        'fraction' : fill_fraction,
        'date' : datetime.today().strftime('%Y-%m-%d')
        })
    hdu.writeto(outname, overwrite=overwrite)


parser = argparse.ArgumentParser()
parser.add_argument('nterms', type=int, help='Number of Zernike modes to generate')
parser.add_argument('mact', type=int, help='Number of actuators across the DM')
parser.add_argument('angle', type=float, help='Incident angle of the beam on the DM (degrees)')
parser.add_argument('fill_fraction', type=float, help='Fraction of the DM filled in the horizontal direction')
parser.add_argument('outname', type=str, help='File to write out')
parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite existing FITS file? Default=False')


if __name__ == '__main__':

    args = parser.parse_args()

    nterms = args.nterms
    nact = args.mact
    angle = args.angle
    fill_fraction = args.fill_fraction 
    outname = args.outname
    overwrite = args.overwrite

    zbasis = zernike_basis(nterms, nact, angle, fill_fraction)
    write_fits(zbasis, outname, nact, angle, fill_fraction, overwrite=overwrite)