import numpy as np


def get_probed_measurements(camstream, dmstream, probe_cmds, navg=1, dmdelay=2):
    
    curcmd = dmstream.grab_latest() # get current cmd on channel
    
    meas = []
    for probe in probe_cmds:
        dmstream.write(probe)
        cnt0 = camstream.md.cnt0
        im = np.mean(camstream.grab_many(navg, cnt0_min=cnt0+dmdelay),axis=0)
        meas.append(im)
        
    dmstream.write(curcmd) # restore DM
    
    return np.asarray(meas)

def get_ref_measurement(camstream, navg=1, dmdelay=2):
    cnt0 = camstream.md.cnt0
    im = np.mean(camstream.grab_many(navg, cnt0_min=cnt0+dmdelay),axis=0)
    return im


def get_response_measurements(camstream, dmstream, dmdivstream, divcmds, modecmds, navg=1, dmdelay=2):

    meas = []
    for cmd in divcmds:

        # write diversity cmd
        dmdivstream.write(cmd)

        # collect interaction measurements for that diversity state
        imcube = get_probed_measurements(camstream, dmstream, modecmds, navg=navg, dmdelay=dmdelay)

        meas.append(imcube)

    return np.asarray(meas).swapaxes(0,1) # nmodes x ndiv x xpix x ypix
