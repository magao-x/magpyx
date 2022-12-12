import numpy as np
from time import sleep

from poppy.zernike import arbitrary_basis
from astropy.io import fits

from purepyindi import INDIClient

from ..imutils import register_images, slice_to_valid_shape, center_of_mass
from ..utils import ImageStream, indi_send_and_wait, str2bool
from ..instrument import move_stage, take_dark

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('fdpr')

def take_measurements_from_config(config_params, dm_cmds=None, client=None, dmstream=None, dmdivstream=None, camstream=None, darkim=None,  restore_dm=True):
    
    skip_indi = config_params.get_param('diversity', 'skip_indi', str2bool)

   # print('INDI STATE ', skip_indi)

    if (client is None) and (not skip_indi):
        # open indi client connection
        client = INDIClient('localhost', config_params.get_param('diversity', 'port', int))
        client.start()

    if dmstream is None:
        # open shmims
        dmstream = ImageStream(config_params.get_param('diversity', 'dmchannel', str))
    if dmdivstream is None:
        dmdivstream = ImageStream(config_params.get_param('diversity', 'dmdivchannel', str))
    if camstream is None:
        camname = config_params.get_param('camera', 'name', str)
        camstream = ImageStream(camname)

    if (darkim is None) and (not skip_indi):
        # take a dark (eventually replace this with the INDI dark [needs some kind of check to see if we have a dark, I guess])
        darkim = take_dark(camstream, client, camname, config_params.get_param('diversity', 'ndark', int))

    dmdelay = config_params.get_param('diversity', 'dmdelay', float)
    indidelay = config_params.get_param('diversity', 'indidelay', float)

    # measure
    div_type = config_params.get_param('diversity', 'type', str)
    if div_type.lower() == 'dm':
        # get defocus mode
        with fits.open(config_params.get_param('interaction', 'dm_mask', str)) as f:
            dm_mask = f[0].data
        zbasis = arbitrary_basis(dm_mask, nterms=4, outside=0)
        defocus_mode = zbasis[-1]
        imcube = measure_dm_diversity(camstream,
                                      dmstream,
                                      dmdivstream,
                                      defocus_mode,
                                      config_params.get_param('diversity', 'values', float),
                                      config_params.get_param('diversity', 'navg', float),
                                      darkim=darkim,
                                      dm_cmds=dm_cmds,
                                      dmdelay=dmdelay,
                                      indidelay=indidelay,
                                      restore_dm=restore_dm
                                      )
    elif div_type.lower() == 'stage': # stage diversity
        positions = np.asarray(config_params.get_param('diversity', 'values', float)) + config_params.get_param('diversity', 'stage_focus', float)
        imcube = measure_stage_diversity(client,
                                camstream,
                                dmstream,
                                config_params.get_param('diversity', 'camstage', str),
                                positions,
                                config_params.get_param('diversity', 'navg', float),
                                darkim=darkim,
                                dm_cmds=dm_cmds,
                                dmdelay=dmdelay,
                                final_position=positions[0],
                                restore_dm=restore_dm
                                )
    elif div_type.lower() == 'camera':
        raise NotImplementedError('multi-camera diversity not implemented yet!')

    print(imcube.shape)
    # clip if needed
    shape = imcube.shape
    naxes = len(shape)
    N = config_params.get_param('estimation', 'N', int)
    if N < shape[-1]: # assume the camera image is square
        logger.info(f'Expected shape {N}x{N} but got shape {shape[-2:]}. Clipping to {N}x{N} about center of mass.')
        imcube_reduced = []
        for im in imcube:
            if naxes == 4:
                # in this case, im is actually a cube
                # so define a slice around the mean center of mass
                # (e.g., response matrix measurements)
                im0 = np.mean(im,axis=0)
                com = center_of_mass(im0)
                totalslice = (slice(None),) + slice_to_valid_shape(im0, com, N, return_slice=True)
                imcube_reduced.append(im[totalslice])
            elif naxes == 3:
                # here, im is actually an image
                # (e.g., measurements for a single estimate)
                com = center_of_mass(im)
                newim = slice_to_valid_shape(im, com, N)
                imcube_reduced.append(newim)
        imcube = np.asarray(imcube_reduced)
    if N > shape[-1]: # assume the camera image is square
        logger.warning(f'Camera frames are smaller than expected. Expected {N}x{N} but got {shape[-2:]}.')

    return imcube

def measure_dm_diversity(camstream, dmstream, dmdivstream, defocus_mode, defocus_vals, nimages, dm_cmds=None, restore_dm=True, dmdelay=None, indidelay=None, improc='mean', darkim=None):

    # get the initial defocus set on the DM
    #client.wait_for_properties([f'{device}.current_amps',])
    #defocus0 = client[f'{device}.current_amps.0002']
    divcmd = dmdivstream.grab_latest()
    
    # commanding DM
    dm_shape = dmstream.grab_latest().shape
    dm_type = dmstream.buffer.dtype
    # keep track of channel cmd
    if restore_dm:
        curcmd = dmstream.grab_latest()
    else:
        curmd = np.zeros(dm_shape)

    if darkim is None:
        darkim = 0

    allims = []
    for j, curdefocus in enumerate(defocus_vals):
        print(f'Moving to focus position {j+1}')
                                
        # send INDI command to apply defocus to DM
        #client[f'{device}.current_amps.0002'] = defocus0 + curdefocus
        dmdivstream.write(defocus_mode*curdefocus + divcmd)
        if indidelay is not None:
            sleep(indidelay)

        # loop over DM commands, and take measurements
        curims = []
        if dm_cmds is None:
            dm_cmds = [np.zeros(dm_shape, dtype=dm_type) + curcmd,]
        for cmd in dm_cmds:
            dmstream.write(cmd.astype(dm_type))
            cnt0 = camstream.md.cnt0 # grab the current camera frame number
            if dmdelay is not None:
                #sleep(dmdelay)
                newcnt0 = cnt0 + dmdelay # expected camera frame number for this DM command
            else:
                newcnt0 = None # don't wait otherwise
            imlist = np.asarray(camstream.grab_many(nimages, cnt0_min=newcnt0))
            if improc == 'register':
                im = np.mean(register_images(imlist - darkim, upsample=10), axis=0)
            else:
                im = np.mean(imlist, axis=0) - darkim
            curims.append(im)
        allims.append(curims)     
        
    # set defocus back to the starting point
    if restore_dm:
        dmstream.write(curcmd)
    #client[f'{device}.current_amps.0002'] = defocus0
    dmdivstream.write(divcmd)
    if indidelay is not None:
        sleep(indidelay)
    return np.squeeze(allims)

def measure_stage_diversity(client, camstream, dmstream, camstage, defocus_positions, nimages, final_position=None, dm_cmds=None, restore_dm=True, dmdelay=None, improc='mean', darkim=None):
    dm_shape = dmstream.grab_latest().shape
    dm_type = dmstream.buffer.dtype
    
    # keep track of channel cmd
    if restore_dm:
        curcmd = dmstream.grab_latest()
    else:
        curcmd = np.zeros(dm_shape)

    if darkim is None:
        darkim = 0

    allims = []
    for j, pos in enumerate(defocus_positions):
        print(f'Moving to focus position {j+1}')
                                
        # block until stage is in position
        sleep(0.5)
        move_stage(client, camstage, pos, block=True)
        sleep(0.1)

        # loop over DM commands, and take measurements
        curims = []
        send_cmd = True
        if dm_cmds is None:
            dm_cmds = [np.zeros(dm_shape, dtype=dm_type) + curcmd,]
            send_cmd = False
        for cmd in dm_cmds:

            if send_cmd:
                dmstream.write(cmd.astype(dm_type))
                dmstream.write(cmd.astype(dm_type)) # for good measure
                cnt0 = camstream.md.cnt0 # grab the current camera frame number
                if dmdelay is not None:
                    #sleep(dmdelay)
                    newcnt0 = cnt0 + dmdelay # expected camera frame number for this DM command
                else:
                    newcnt0 = None # don't wait otherwise
            else:
                newcnt0 = None
            imlist = np.asarray(camstream.grab_many(nimages, cnt0_min=newcnt0))
            if improc == 'register':
                im = np.mean(register_images(imlist - darkim, upsample=10), axis=0)
            else:
                im = np.mean(imlist, axis=0) - darkim
            curims.append(im)
        allims.append(curims)  
        
        if restore_dm and send_cmd:
            dmstream.write(curcmd.astype(dm_type)) # reset between stage moves (minimize creep on ALPAOs)  
        
    # restore
    if restore_dm and send_cmd:
        dmstream.write(curcmd)
    if final_position is not None:
        move_stage(client, camstage, final_position, block=False)
    return np.squeeze(allims)


def measure_multicam_stage_diversity(client, camstream1, camstream2, dmstream, camstage1, camstage2, defocus_positions, nimages, final_positions=None, dm_cmds=None, restore_dm=True, dmdelay=None, improc='mean', darkims=None):
    '''
    Currently assuming just two cameras (2 diversity measurements)

    TO DO:
    * add support for multiple dark images
    * multiple stage focus positions
    * multiple stage defocus positions
    * ImageStream.grab_many blocks, so we're technically always getting a measurement from one camera sooner than on the other (would need to thread)
    * figure out right output shape
    '''
   
    dm_shape = dmstream.grab_latest().shape
    dm_type = dmstream.buffer.dtype
    
    # keep track of channel cmd
    if restore_dm:
        curcmd = dmstream.grab_latest()
    else:
        curcmd = np.zeros(dm_shape)

    if darkims is None:
        darkim1 = darkim2 = 0

    # move stages (current)
    move_stage(client, camstage1, defocus_positions[0], block=True)
    move_stage(client, camstage2, defocus_positions[1], block=True)
    sleep(0.1)
    
    # loop over DM commands, and take measurements
    curims1 = []
    curims2 = []
    send_cmd = True
    if dm_cmds is None:
        dm_cmds = [np.zeros(dm_shape, dtype=dm_type) + curcmd,]
        send_cmd = False
    for cmd in dm_cmds:

        if send_cmd:
            dmstream.write(cmd.astype(dm_type))
            dmstream.write(cmd.astype(dm_type)) # for good measure

            cnt0_1 = camstream1.md.cnt0 # grab the current camera frame number
            cnt0_2 = camstream2.md.cnt0
            if dmdelay is not None:
                #sleep(dmdelay)
                newcnt0_1 = cnt0_1 + dmdelay # expected camera frame number for this DM command
                newcnt0_2 = cnt0_2 + dmdelay
            else:
                newcnt0_1 = newcnt0_2 = None # don't wait otherwise
        else:
            newcnt0_1 = newcnt0_2 = None # don't wait if no command is sent

        # non-simultaneous! FIX ME
        imlist1 = np.asarray(camstream1.grab_many(nimages, cnt0_min=newcnt0_1)) 
        imlist2 = np.asarray(camstream2.grab_many(nimages, cnt0_min=newcnt0_2))

        if improc == 'register':
            im1 = np.mean(register_images(imlist1 - darkims[0], upsample=10), axis=0)
            im2 = np.mean(register_images(imlist2 - darkims[1], upsample=10), axis=0)
        else:
            im1 = np.mean(imlist1, axis=0) - darkims[0]
            im2 = np.mean(imlist1, axis=0) - darkims[1]
        curims1.append(im1)
        curims2.append(im2)

    allims = np.concatenate([im1, im2], axis=0) # not sure about the correct shape 

    if restore_dm and send_cmd:
        dmstream.write(curcmd.astype(dm_type)) # reset between stage moves (minimize creep on ALPAOs)  
    if final_positions is not None:
        move_stage(client, camstage1, final_positions[0], block=False)
        move_stage(client, camstage2, final_positions[1], block=False)

    return np.squeeze(allims)