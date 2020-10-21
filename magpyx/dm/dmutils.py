'''
Various functions frequently used in DM analysis, closed loop, etc.
'''
import numpy as np
from itertools import product
from scipy.linalg import hadamard
from skimage import draw
from ..imutils import rot_matrix, rotate

def get_alpao_actuator_coords(D_pupil, rotation=0, extra_scaling= (1., 1.), offset=(0,0)):
    # define (y, x) coords for actuators
    # recall that actuators extend outside the clear aperture (see Rooms et al. 2010)
    act_pitch = 1.5e-3
    D_alpao = 13.5e-3
    n_act = 11
    act_spacing = np.linspace(n_act/2.*act_pitch, -n_act/2.*act_pitch, num=n_act)
    yx = np.asarray(list(product(act_spacing, -act_spacing)))
    
    r = np.sqrt(yx[:,0]**2 + yx[:,1]**2)
    yx = yx[r <= (D_alpao+act_pitch*3)/2.]
    
    # scale based on beam footprint (to first approx, pupil_diam)
    scale = D_pupil / D_alpao
    yx *= scale * np.asarray(extra_scaling)
    
    # apply rotation
    y, x = rotate(yx[:,0], yx[:,1], rotation)
    return y + offset[0], x + offset[1]


def get_bmc_actuator_coords(D_pupil, rotation=0, extra_scaling= (1., 1.), offset=(0,0)):
    # define (y, x) coords for actuators
    # recall that actuators extend outside the clear aperture (see Rooms et al. 2010)
    act_pitch = 0.4e-3 #mm
    D_bmc = 19.6e-3
    n_act = 50
    act_spacing = np.linspace(n_act/2.*act_pitch, -n_act/2.*act_pitch, num=n_act)
    yx = np.asarray(list(product(act_spacing, -act_spacing)))
    
    r = np.sqrt(yx[:,0]**2 + yx[:,1]**2)
    yx = yx[r <= (D_bmc+act_pitch*3.2)/2.]
    
    # scale based on beam footprint (to first approx, pupil_diam)
    scale = D_pupil / D_bmc
    yx *= scale * np.asarray(extra_scaling)
    
    # apply rotation
    y, x = rotate(yx[:,0], yx[:,1], rotation)
    return y + offset[0], x + offset[1]

def map_square_to_vector(cmd_square, dm_map, dm_mask):
    '''DM agnostic mapping function'''
    filled_mask = dm_map != 0
    nact = np.count_nonzero(filled_mask)
    vec = np.zeros(nact, dtype=cmd_square.dtype)
    
    mapping = dm_map[dm_mask] - 1
    vec[mapping] = cmd_square[dm_mask]
    return vec

def map_vector_to_square(cmd_vec, dm_map, dm_mask):
    '''DM agnostic mapping function'''
    filled_mask = dm_map != 0
    sq = np.zeros(dm_map.shape, dtype=cmd_vec.dtype)
    
    mapping = dm_map[filled_mask] - 1
    sq[filled_mask] = cmd_vec[mapping]
    sq *= dm_mask
    return sq

def select_actuators_from_command(act_y, act_x, cmd, dm_map, dm_mask):
    '''
    Given a (binary) command, return the actuator positions
    this corresponds to
    '''
    cmd_vec = map_square_to_vector(cmd, dm_map, dm_mask.astype(bool))
    order = map_square_to_vector(dm_map*cmd.astype(bool), dm_map, dm_mask.astype(bool))[cmd_vec.astype(bool)]
    return act_y[cmd_vec.astype(bool)], act_x[cmd_vec.astype(bool)], order

def get_hadamard_modes(dm_mask, roll=0, shuffle=None):
    nact = np.count_nonzero(dm_mask)
    if shuffle is None:
        shuffle = slice(nact)
    np2 = 2**int(np.ceil(np.log2(nact)))
    print(f'Generating a {np2}x{np2} Hadamard matrix.')
    hmat = hadamard(np2)
    return np.roll(hmat[shuffle,:nact], roll, axis=1)
    '''cmds = []
    nact = np.count_nonzero(dm_mask)
    for n in range(nact):
        cmd = np.zeros(nact)
        cmd[n] = 1
        cmds.append(cmd)
    return np.asarray(cmds)'''

def find_nearest(slaved_vec_idx, slaved_map, dm_map, dm_mask, n=1):
    
    # loop over slaved actuators
    neighbors = []
    for idx in slaved_vec_idx:
        
        # find actuator 2D map index 
        vec0 = np.zeros(2040)
        vec0[idx] = 1
        map0 = map_vector_to_square(vec0, dm_map, dm_mask)
        actyx = np.squeeze(np.where(map0.astype(bool)))
    
        # for each, get a distance map
        indices = np.indices((50,50))
        indices[0] -= actyx[0]
        indices[1] -= actyx[1]
        
        dist = np.sqrt(indices[0]**2 + indices[1]**2)
        dsort = np.unravel_index(np.argsort(np.ma.masked_where(slaved_map|~dm_mask.astype(bool), dist), axis=None), dist.shape)
        neighbor_map = np.zeros_like(map0)
        neighbor_map[dsort[0][:n], dsort[1][:n]] = 1
        
        neighbors.append(np.squeeze(np.where(map_square_to_vector(neighbor_map, dm_map, dm_mask.astype(bool)))))

    return neighbors

def get_slave_map(ifmat, threshold, dm_mask, map_vec_to_square_func, map_square_to_vec_func):
    
    if_rms = np.sqrt(np.mean(ifmat**2,axis=(1)))
    bad = map_vec_to_square_func(if_rms) < threshold
    slaved = bad & dm_mask

    slaved_vec = map_vec_to_square_func(slaved)
    good_vec = map_square_to_vec_func(~slaved)
    slaved_vec_idx = np.where(slaved_vec)[0]

    ifs_good = ifmat[good_vec]
    
    return slaved_vec, slaved_vec_idx, slaved, ifs_good

def fill_in_slaved_cmds(cmd_vec, slaved_vec_idx, neighbor_mapping):
    cmd = cmd_vec.copy()
    for slaved, neighbors in zip(slaved_vec_idx, neighbor_mapping):
        cmd[slaved] = np.mean(cmd[neighbors])
    return cmd

def plop_down_a_mask_on_a_location(y, x, r, shape):
    mask = np.zeros(shape, dtype=bool)
    idx = draw.circle(y, x, r, shape)
    mask[idx] = 1
    return mask

def remove_lo_from_if(image, mask, zbasis):
    im = image
    act_loc = np.where(im*mask == (im*mask).min())
    circ_idx = draw.circle(act_loc[0][0], act_loc[1][0], 15, shape=image.shape)
    circ_mask = np.ones_like(mask)
    circ_mask[circ_idx] = 0
    tot_mask = mask & circ_mask
    im_planerem = remove_plane(im, tot_mask) * mask
    
    zcoeffs = zernike.opd_expand(im_planerem, aperture=tot_mask, nterms=len(zbasis), basis=get_zbasis)
    return im_planerem - zernike.opd_from_zernikes(zcoeffs, basis=get_zbasis, aperture=tot_mask, outside=0)

def get_distance(locyx, dm_map, dm_mask):
    idy, idx = np.indices(dm_mask.shape)
    idy -= locyx[0]
    idx -= locyx[1]
    distance = np.sqrt(idy**2 + idx**2)
    return map_square_to_vector(distance, dm_map, dm_mask.astype(bool))

def get_grid_cmds(dm_shape, ngrid, val, do_plusminus=True):

    grid_cmds = []
    for n in range(ngrid):
        for m in range(ngrid):
            cmd = np.zeros(dm_shape)
            cmd[n::ngrid,m::ngrid] = val
            grid_cmds.append(cmd)
    grid_cmds = np.asarray(grid_cmds)
    
    if do_plusminus:
        allcmds = np.vstack([grid_cmds, -grid_cmds])
    else:
        allcmds = np.asarray(grid_cmds)

    return allcmds