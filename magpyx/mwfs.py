import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.dpi' : 100,
    'image.origin' : 'lower',
    'font.size' : 14
})

import numpy as np
from astropy.io import fits
from .utils import ImageStream

from purepyindi import INDIClient

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('mwfs')

# camsci2 spots relative to leak
# probably don't want this hard-coded
allspots_camsci2 = [
    (53, -50),
    (150, -143),
    (149, -107),
    (148, -70),
    (113, -144),
    (112, -108),
    (111, -71),
    (76, -145),
    (76, -108),
    (-54, 50),
    (-150, 142),
    (-149, 106),
    (-148, 70),
    (-113, 143),
    (-112, 107),
    (-111, 70),
    (-77, 144),
    (-76, 108)]

def cut_out(image, cenyx, boxsize):
    subslice = (slice(cenyx[0] - boxsize//2, cenyx[0] + boxsize//2),
                slice(cenyx[1] - boxsize//2, cenyx[1] + boxsize//2))
    return image[subslice]

def locate_peak(image):
    return tuple(np.argwhere(image == image.max())[0])

def subcoord_to_imcoord(subcenyx, subsize, peakcenyx):
    return (subcenyx[0] - subsize//2 + peakcenyx[0], subcenyx[1] - subsize//2 + peakcenyx[1])

def find_spots_on_camera(leak_guess, wavelength=656.28, camera='camsci2', leak_search_box=80, refine_spots=False, spot_search_box=50):

    # grab an image
    camera = ImageStream(camera)
    im = camera.grab_latest()
    camera.close()

    # refine leak location
    leak_coords = subcoord_to_imcoord(leak_guess, leak_search_box, locate_peak(cut_out(im, leak_guess, leak_search_box)))
    logger.info(f'Refined leak coords from {(leak_guess[0], leak_guess[1])} to {(leak_coords[0], leak_coords[1])}')

    # refine relative + scaled leak coords
    scale = wavelength / 656.28 # relative to Halpha
    spot_guess = [( int(np.rint(s[0] * scale + leak_coords[0])), int(np.rint(s[1] * scale + leak_coords[1]))) for s in allspots_camsci2]
    if refine_spots:
        spot_coords = [subcoord_to_imcoord(s, spot_search_box, locate_peak(cut_out(im, s, spot_search_box))) for s in spot_guess]
        for sg, sr in zip(spot_guess, spot_coords):
            logger.info(f'Refined MWFS spot coords from {(sg[0], sg[1])} to {(sr[0], sr[1])}')

    else:
        spot_coords = spot_guess

    return spot_coords

def display_spots_on_camera(camera, spot_locs, nimages=1, vmin=None, vmax=None):

    # grab an image
    camera = ImageStream(camera)
    im = np.mean(camera.grab_many(nimages),axis=0)
    camera.close()

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    mim = ax.imshow(im, vmin=vmin, vmax=vmax)

    for s in spot_locs:
        plt.scatter(s[1], s[0], marker='x', c='C3')

    return mim, fig, ax
    
def move_spots_to_indi(spot_locs):

    client = INDIClient('localhost', 7624)
    client.start()

    for n, s in enumerate(spot_locs):
        cmd_dict = {
            f'mwfsMonitor.xSpot.{n:0>2}': {'value': s[0]},
            f'mwfsMonitor.ySpot.{n:0>2}': {'value': s[1]},
        }
        client.wait_for_state(cmd_dict, wait_for_properties=True, timeout=10)
    logger.info("Moved spot locations to mwfsMonitor.")
    