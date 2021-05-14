'''
What goes here?

Two version of the measurement function:
* w/ stage diversity (generalize to no DM)
* w/ DM diversity
    - currently uses dmModes to get best defocus mode. Not sure how ideal this approach is.

Is that it?
'''
import numpy as np
from time import sleep

from purepyindi import INDIClient

from ..imutils import register_images
from ..utils import ImageStream, indi_send_and_wait
from ..instrument import move_stage, take_dark


def take_measurements_from_config(config_params, dm_cmds=None, delay=None):

    # open indi client connection
    client = INDIClient('localhost', config_params.get_param('diversity', 'port', int))
    client.start()

    # open shmims
    dmstream = ImageStream(config_params.get_param('diversity', 'dmdivchannel', str))
    camname = config_params.get_param('camera', 'name', str)
    camstream = ImageStream(camname)

    # take a dark (eventually replace this with the INDI dark [needs some kind of check to see if we have a dark, I guess])
    darkim = take_dark(camstream, client, camname, config_params.get_param('diversity', 'ndark', int))

    # measure
    div_type = config_params.get_param('diversity', 'type', str)
    if div_type.lower() == 'dm':
        imcube = measure_dm_diversity(client,
                                      config_params.get_param('diversity', 'dmModes', str),
                                      camstream,
                                      dmstream,
                                      config_params.get_param('diversity', 'values', float),
                                      config_params.get_param('diversity', 'navg', float),
                                      darkim=darkim,
                                      dm_cmds=dm_cmds,
                                      delay=delay
                                      )
    else: # stage diversity
        imcube = measure_stage_diversity(client,
                                camstream,
                                dmstream,
                                config_params.get_param('diversity', 'camstage', str),
                                config_params.get_param('diversity', 'values', float),
                                config_params.get_param('diversity', 'navg', float),
                                darkim=darkim,
                                dm_cmds=dm_cmds,
                                delay=delay
                                )
    return imcube

def measure_dm_diversity(client, device, camstream, dmstream, defocus_vals, nimages, dm_cmds=None, zero_dm=True, delay=None, improc='mean', darkim=None):

    # get the initial defocus set on the DM
    client.wait_for_properties([f'{device}.current_amps',])
    defocus0 = client[f'{device}.current_amps.0002']
    
    # commanding DM
    dm_shape = dmstream.grab_latest().shape
    dm_type = dmstream.buffer.dtype
    # zero out the DM if requested
    if zero_dm:
        dmstream.write(np.zeros(dm_shape).astype(dm_type))
    
    if darkim is None:
        darkim = 0

    allims = []
    for j, curdefocus in enumerate(defocus_vals):
        print(f'Moving to focus position {j+1}')
                                
        # send INDI command to apply defocus to DM
        client[f'{device}.current_amps.0002'] = defocus0 + curdefocus
        sleep(1.0)

        # loop over DM commands, and take measurements
        curims = []
        if dm_cmds is None:
            dm_cmds = [np.zeros(dm_shape, dtype=dm_type),]
        for cmd in dm_cmds:
            dmstream.write(cmd.astype(dm_type))
            if delay is not None:
                sleep(delay)
            imlist = np.asarray(camstream.grab_many(nimages))
            if improc == 'register':
                im = np.mean(register_images(imlist - darkim, upsample=10), axis=0)
            else:
                im = np.mean(imlist, axis=0) - darkim
            curims.append(im)
        allims.append(curims)     
        
    # set defocus back to the starting point
    if zero_dm:
        dmstream.write(np.zeros(dm_shape).astype(dmstream.buffer.dtype))
    client[f'{device}.current_amps.0002'] = defocus0
    sleep(1.0)
    return np.squeeze(allims)


def measure_stage_diversity(client, camstream, dmstream, camstage, defocus_positions, nimages, final_position=None, dm_cmds=None, zero_dm=True, delay=None, improc='mean', darkim=None):
    dm_shape = dmstream.grab_latest().shape
    dm_type = dmstream.buffer.dtype
    
    # zero out the DM if requested
    if zero_dm:
        dmstream.write(np.zeros(dm_shape).astype(dm_type))

    if darkim is None:
        darkim = 0

    allims = []
    for j, pos in enumerate(defocus_positions):
        print(f'Moving to focus position {j+1}')
                                
        # block until stage is in position
        move_stage(client, camstage, pos, block=True)

        # loop over DM commands, and take measurements
        curims = []
        if dm_cmds is None:
            dm_cmds = [np.zeros(dm_shape, dtype=dm_type),]
        for cmd in dm_cmds:
            dmstream.write(cmd.astype(dm_type))
            if delay is not None:
                sleep(delay)
            imlist = np.asarray(camstream.grab_many(nimages))
            if improc == 'register':
                im = np.mean(register_images(imlist - darkim, upsample=10), axis=0)
            else:
                im = np.mean(imlist, axis=0) - darkim
            curims.append(im)
        allims.append(curims)      
        
    # restore
    if zero_dm:
        dmstream.write(np.zeros(dm_shape).astype(dmstream.buffer.dtype))
    if final_position is not None:
        move_stage(client, camstage, final_position, block=False)
    return np.squeeze(allims)