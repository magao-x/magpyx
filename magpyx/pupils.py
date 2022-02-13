'''
Module for representations of bump mask, open pupil, asymmetric spider, and vAPP

To do:
* document
* clean up variable names
* finish implementing pupils
'''

import numpy as np
from skimage import draw
from .imutils import rot_matrix, rotate

def get_coronagraphic_pupil(delta_phys, N, extra_rot=0):
    '''
    At the 9 mm diameter pupil:
    * Mask outer diameter: 8.6040 mm
    * Mask inner diameter: 2.7900 mm
    * Spider origin: 0.4707 mm above and below the center (before rotation)
    * Spider rotation: 38.75 deg c.c.w. to mask bump.
    * Spider width: 0.1917 mm
    * Bump mask diameter: 0.5742 mm
    * Bump mask position: (2.853 mm, -0.6705 mm) relative to the center. 
    '''
    
    pupil_analytic = np.zeros((N,N))
    cen = (N-1)/2.
    
    # inner and outer diameters
    D = 8.604e-3
    D_sm = 2.79e-3
    
    Didx = draw.circle(cen, cen, D/delta_phys/2, shape=(N,N))
    Dsm_idx = draw.circle(cen, cen, D_sm/delta_phys/2, shape=(N,N))
    
    pupil_analytic[Didx] = 1
    pupil_analytic[Dsm_idx] = 0
    
    #spider arms
    spider_base = 0.4707e-3 / delta_phys
    spider_halfwidth = 0.1917e-3 / delta_phys / 2
    spider_top = N
    rots = (-45,45)
    global_rot = 38.75+extra_rot
    for rot in rots:
    
        cx = (cen-spider_halfwidth, cen-spider_halfwidth, cen+spider_halfwidth, cen+spider_halfwidth)
        cy = (spider_base+cen, spider_top, spider_top, spider_base+cen)
        cyr, cxr = rotate(cy, cx, rot, ceny=cen+spider_base, cenx=cen)
        cyr, cxr = rotate(cyr, cxr, global_rot, ceny=cen, cenx=cen)
        recidx = draw.polygon(cyr,cxr, shape=(N,N))
        pupil_analytic[recidx] = 0
        
        cy = (cen-spider_base, 0, 0, cen-spider_base)
        cyr, cxr = rotate(cy, cx, rot, ceny=cen-spider_base, cenx=cen)
        cyr, cxr = rotate(cyr, cxr, global_rot, ceny=cen, cenx=cen)
        recidx = draw.polygon(cyr,cxr, shape=(N,N))
        pupil_analytic[recidx] = 0

    # add the bump
    by = cen-0.6705e-3 / delta_phys
    bx = cen+2.853e-3 / delta_phys
    br = 0.5742e-3 / delta_phys / 2.
    byr, bxr = rotate([by,], [bx,], extra_rot, ceny=cen, cenx=cen)
    bidx = draw.circle(byr[0], bxr[0], br, shape=(N,N))
    pupil_analytic[bidx] = 0
    
    return pupil_analytic

def get_open_pupil(delta_phys, N, extra_rot=0):
    # rotation argument is meaningless here, but included for consistency with the other pupil functions
    
    pupil_analytic = np.zeros((N,N))
    cen = (N-1)/2.
    
    # inner and outer diameters
    D = 9e-3
    D_sm = 0.3 * D #0.239 * D
    
    Didx = draw.circle(cen, cen, D/delta_phys/2, shape=(N,N))
    Dsm_idx = draw.circle(cen, cen, D_sm/delta_phys/2, shape=(N,N))
    
    pupil_analytic[Didx] = 1
    pupil_analytic[Dsm_idx] = 0
    
    return pupil_analytic