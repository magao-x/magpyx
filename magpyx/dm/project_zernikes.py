#!/usr/bin/env python

from datetime import datetime
import numpy as np
from astropy.io import fits

def projected_basis(nterms, angle, fill_fraction, actuator_mask):
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
        angle : float
            Incidence angle of the beam on the DM, given in
            degrees. Angles are always assumed to be in the x-z
            plane.
        fill_fraction : float
            Fraction of the DM in the beam footprint (defined
            along the x axis) [=0.96 for the 2K]
        actuator_mask : str
            Path to FITS file with actuators mapped from 1 to n actuators.

    Returns: list of nterms modes of shape nact x act 
    '''
    from poppy.zernike import zernike_basis, arbitrary_basis

    if isinstance(actuator_mask, str):
        with fits.open(actuator_mask) as f:
            dm_mask = f[0].data.astype(bool)
    else:
        dm_mask = actuator_mask

    nact = dm_mask.shape[0]

    # generate the radial and theta arrays that account for geometrical
    # projection
    cos = np.cos(np.deg2rad(angle))
    x = np.linspace(-(nact-1)/nact, (nact-1)/nact, num=nact, endpoint=True) / fill_fraction
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx**2 + (yy/cos)**2)
    theta = np.arctan2(yy/cos, xx)

    # make the modes
    #zbasis = zernike_basis(nterms=nterms, npix=nact, outside=0., rho=rho, theta=theta) 
    # generate nterms+1 because we're throwing away piston later
    zbasis = arbitrary_basis(aperture=rho <= 1.0, nterms=nterms+1, outside=0., rho=rho, theta=theta) 

    # find actuators outside the beam footprint
    footprint = zbasis[0] != 0
    outside = dm_mask & ~footprint
    outyx = np.dstack(np.where(outside))[0]

    # interpolate from nearest neighbors
    for y, x in outyx:
        yneighbor, xneighbor = find_nearest((y, x), footprint, num=1)
        zbasis[:, y, x] = zbasis[:, yneighbor, xneighbor]

    # toss piston
    return zbasis[1:]

def find_nearest(yx, footprint, num=1):
    '''
    Return the nearest neighbor(s) of an actuator
    within the beam footprint
    '''
    
    act_pos = yx
    indices = np.indices(footprint.shape)
    
    distance = np.sqrt( (indices[0]-act_pos[0])**2 + (indices[1]-act_pos[1])**2 )
    
    dist_sort = np.argsort(np.ma.masked_where(~footprint, distance), axis=None)
    closest_idx = dist_sort[:num]
    closest = (indices[0].flat[closest_idx][0], indices[1].flat[closest_idx][0])
    
    return closest

def write_fits(zbasis, outname, angle, fill_fraction, overwrite=False):
    '''Write zernike basis out to FITS file'''

    hdu = fits.PrimaryHDU(np.asarray(zbasis, dtype=np.float32))
    hdu.header.update({
        #'nact' : nact,
        'angle' : angle,
        'fraction' : fill_fraction,
        'date' : datetime.today().strftime('%Y-%m-%d')
        })
    hdu.writeto(outname, overwrite=overwrite)

def main():

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('nterms', type=int, help='Number of Zernike modes to generate')
    parser.add_argument('angle', type=float, help='Incident angle of the beam on the DM (degrees)')
    parser.add_argument('fill_fraction', type=float, help='Fraction of the DM filled in the horizontal direction')
    parser.add_argument('actuator_mask', type=str, help='Path to FITS file with binary DM actuator mask (probably under /opt/MagAOX/calib/dm/[dm_name]/')
    parser.add_argument('outname', type=str, help='File to write out')
    parser.add_argument('--overwrite', type=bool, default=False, help='Overwrite existing FITS file? Default=False')

    args = parser.parse_args()

    nterms = args.nterms
    angle = args.angle
    fill_fraction = args.fill_fraction 
    actuator_mask = args.actuator_mask
    outname = args.outname
    overwrite = args.overwrite

    zbasis = projected_basis(nterms, angle, fill_fraction, actuator_mask)
    write_fits(zbasis, outname, angle, fill_fraction, overwrite=overwrite)


if __name__ == '__main__':
    main()