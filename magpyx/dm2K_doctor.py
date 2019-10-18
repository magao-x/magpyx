import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
    'image.origin' : 'lower',
    'figure.dpi' : 100,
}
plt.rcParams.update(params)

from astropy.io import fits
import numpy as np
from skimage import draw

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('dm_doctor')

def colorbar(mappable, *args, **kwargs):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax, *args, **kwargs)

def map_square_to_vector_2K(array):
    '''
    Given the dm data values embedded
    in a square (50x50) array, pull out
    the actuator values in an properly ordered
    vector.
    Parameters:
        array : nd array
            2D (50x50) array of DM inputs
    Returns:
        vector : nd array
            2040-element input vector
    '''
    mask = mask_2K()
    return (array[::-1,::-1].T)[mask]

def map_vector_to_square_2K(vector):
    '''
    Given the DM data values in a vector
    ordered by actuator number, embed the
    data in a square array.
    Parameters:
        vector : array-like
            2040-element DM input to be embedded
            in 50x50 square array.
    Returns:
        array : nd array
            50x50 square array
    '''
    array = np.zeros((50,50))
    mask = mask_2K()
    array[mask] = vector
    return array[::-1,::-1].T

def map_index1d_to_index2d_2K(index1d):
    arr = np.zeros(2040)
    arr[index1d] = 1.0
    return np.where(map_vector_to_square_2K(arr))

def map_index2d_to_index1d_2K(index2d):
    arr = np.zeros((50,50))
    arr[index2d] = 1.0
    return np.where(map_square_to_vector_2K(arr))


def mask_2K():
    mask = np.zeros((50,50), dtype=bool)
    circmask = draw.circle(24.5,24.5,25.6,(50,50))
    mask[circmask] = 1
    return mask

def actuator_mapping_2K():
    order = range(2040)
    return map_vector_to_square_2K(order)

def get_rms_map_2K(zrespM):

    if isinstance(zrespM, str):
        with fits.open(zrespM) as f:
            data = f[0].data
    else:
        data = zrespM

    return np.sqrt(np.mean(data**2, axis=(1,2))).reshape((50, 50))

def read_actuator_connection_mapping(filename='/opt/MagAOX/calib/dm/bmc_2k/actuator_mapping_BMC2K_2019_09_27.csv',
                                     return_dump=False):
    actuator_dump = np.genfromtxt(filename, delimiter=',',skip_header=1,dtype=None, encoding=None,
                                 names=['actuator', 'row', 'col', 'bondpad chip', 'bondpad awb',
                                        'awb side', 'bond finger', 'megarray', 'samtec'])
    if not return_dump:
        return np.array([a['megarray'][0] + '-' + a['samtec'][:2] for a in actuator_dump ])
    else:
        return actuator_dump

def format_mpl_coord_2K(ax):
    def format_coord(x, y):
        xint = int(np.rint(x))
        yint = int(np.rint(y))
        return f'x={xint}, y={yint}, actuator number={map_index2d_to_index1d_2K((yint, xint))[0][0]}'
    ax.format_coord = format_coord

def format_mpl_connection_2K(im, conn_types):
    def format_cursor_data(data):
        val = int(data) -1
        if val < 0: return 'None'
        return f', connection={conn_types[val]}'
    im.format_cursor_data = format_cursor_data

def display_actuator_connections(filename=None):
    act_mapping = read_actuator_connection_mapping(filename=filename)
    numeric_mapping = np.zeros(2040)
    conn_types = np.unique(act_mapping)
    for i, c in enumerate(conn_types):
        numeric_mapping[act_mapping == c] = i + 1
        
    cmap = plt.cm.get_cmap('Set2', len(conn_types))
    cmap.set_under([0.5, 0.5, 0.5])

    fig, ax = plt.subplots(1,1, figsize=(6,5))
    im = ax.imshow(map_vector_to_square_2K(numeric_mapping), cmap=cmap)
    cbar = fig.colorbar(im, ax=ax, ticks=range(1,len(conn_types)+1))
    im.set_clim(.5, len(conn_types) + .5)
    cbar.set_ticklabels(conn_types)

    format_mpl_coord_2K(ax)
    format_mpl_connection_2K(im, conn_types)

    return fig, ax

def check_actuator_functionality(zrespM, zrespM_ref=None, actuator_mapfile=None, sigma=2, display=False):
    '''
    Flag suspicious actuators.

    Parameters:
        zrespM : str
            Path to a cacao zrespM fits file
        zrespM_ref : str, optional
            Path to a cacao zrespM fits taken with functioning actuators.
            While not required, this code is MUCH better are flagging misbehaving
            actuators if a reference zrespM is given.
        actuator_mapfile: str, opt
            Path to file that maps actuators to connections. Generated from BMC
            excel spreadsheet. Default to one in the 2K calib directory.
        sigma : float, opt
            Flag threshold. Default: 2
        display : bool, opt
            Display functionality plots? Default: False

    Returns: locations of flagged actuators
    '''
    
    # identify "bad" actuators
    rms = get_rms_map_2K(zrespM)
    #rms = rms - np.mean(rms[mask_2K()]) * mask_2K()
    rms /= rms.max()
    
    if zrespM_ref is not None:
        rms_ref = get_rms_map_2K(zrespM_ref)
        rms_ref /= rms_ref.max()
    else:
        rms_ref = np.zeros_like(rms)
        #rms = unsharp_mask(rms - np.mean(rms[mask_2K()]), amount=1, radius=1.0)*mask_2K()
        
    std = np.std((rms-rms_ref)[mask_2K()])
    bad = np.abs(rms-rms_ref) > std*sigma
    bad_mapping = np.where(map_square_to_vector_2K(bad))[0]
        
    # map back to megarray-samtec connection
    actuator_mapping = read_actuator_connection_mapping(actuator_mapfile)
    connection_mapping = actuator_mapping[map_square_to_vector_2K(bad)]
    
    connect_types = np.unique(actuator_mapping)
    logger.warning(f'Possible Bad Connections:')
    logger.warning(f'(Megarray-Samtec: # Flagged)')
    for c in connect_types:
        logger.warning(f'{c}: {np.count_nonzero(connection_mapping == c)}')
    
    if display:
        fig, axes = plt.subplots(2,2, figsize=(10,7))
        
        # display RMS maps
        
        im = axes[0][0].imshow(rms)
        colorbar(im)
        axes[0][0].set_title('RMS Response')
        format_mpl_coord_2K(axes[0][0])
        
        im = axes[0][1].imshow(rms_ref)
        colorbar(im)
        axes[0][1].set_title('RMS Response (Reference)')
        format_mpl_coord_2K(axes[0][1])
        
        #display actuator connections
        numeric_mapping = np.zeros(2040)
        conn_types = np.unique(actuator_mapping)
        for i, c in enumerate(conn_types):
            numeric_mapping[actuator_mapping == c] = i + 1
            
        bad_mapping = np.zeros(2040)
        bad_mapping[map_square_to_vector_2K(bad)] = numeric_mapping[map_square_to_vector_2K(bad)]
        
        cmap = plt.cm.get_cmap('Set2', len(conn_types))
        cmap.set_under([0.5, 0.5, 0.5])
        im = axes[1][0].imshow(map_vector_to_square_2K(bad_mapping), cmap=cmap)

        cbar = colorbar(im, ticks=range(1,len(conn_types)+1))
        im.set_clim(.5, len(conn_types) + .5)
        cbar.set_ticklabels(conn_types)
        format_mpl_coord_2K(axes[1][0])
        format_mpl_connection_2K(im, conn_types)

        bad_cat, bad_count = np.unique(connection_mapping, return_counts=True)

        axes[1][1].bar(bad_cat, bad_count)
        axes[1][1].set_ylabel('N Flagged')

        fig.tight_layout()
        
        return np.where(map_square_to_vector_2K(bad))[0], connection_mapping
    
    return bad_mapping, connection_mapping

def main():
    # parse command line arguments
    import argparse
    import sys

    parser = argparse.ArgumentParser()

if __name__ == '__main__':
    main()