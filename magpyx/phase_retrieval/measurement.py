'''
What goes here?

Two version of the measurement function:
* w/ stage diversity (generalize to no DM)
* w/ DM diversity

Is that it?
'''

from purepyindi import INDIClient
from ..imutils import register_images
from ..utils import ImageStream, indi_send_and_wait
from ..instrument import move_stage


def measure_dm_diversity(client, camstream, dmstream, defocus_vals, nimages, dm_cmds=None, zero_dm=True, delay=None, improc='mean', darkim=None):

    # get the initial defocus set on the DM
    client.wait_for_properties([f'wooferModes.current_amps',])
    defocus0 = client['wooferModes.current_amps.0002']
    
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
        client['wooferModes.current_amps.0002'] = defocus0 + curdefocus
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
    client['wooferModes.current_amps.0002'] = defocus0
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