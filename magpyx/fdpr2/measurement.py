import numpy as np


def get_probed_measurements(camstream, dmstream, probe_cmds, navg=1, dmdelay=2):
    
    curcmd = dmstream.grab_latest() # get current cmd on channel
    
    meas = []
    for probe in probe_cmds:
        dmstream.write(probe)
        cnt0 = camstream.md.cnt0
        im = np.mean(camstream.grab_many(navg, cnt0_min=cnt0+dmdelay),axis=0)
        #im -= np.median(im)
        meas.append(im)
        
    dmstream.write(curcmd) # restore DM
    
    return np.asarray(meas)

def get_ref_measurement(camstream, navg=1, dmdelay=2):
    cnt0 = camstream.md.cnt0
    im = np.mean(camstream.grab_many(navg, cnt0_min=cnt0+dmdelay),axis=0)
    #im -= np.median(im)
    return im


def get_response_measurements(camstream, dmstream, dmdivstream, divcmds, modecmds, navg=1, dmdelay=2):

    cur0 = dmstream.grab_latest()
    cur1 = dmdivstream.grab_latest()

    meas = []
    for i, cmd in enumerate(modecmds):
        if (i % 100) == 0:
            print(f'{i+1}/{len(modecmds)}')

        # write modal cmd
        dmstream.write(cmd)

        # collect diversity measurements for that mode
        imcube = get_probed_measurements(camstream, dmdivstream, divcmds, navg=navg, dmdelay=dmdelay)

        meas.append(imcube)

    # restore
    dmstream.write(cur0)
    dmdivstream.write(cur1)

    return np.asarray(meas)#.swapaxes(0,1) # nmodes x ndiv x xpix x ypix
