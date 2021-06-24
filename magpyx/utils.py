import purepyindi as indi
import numpy as np
import ImageStreamIOWrap as shmio
from time import sleep

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils')

def indi_send_and_wait(client, cmd_dict, tol=1e-3, wait_for_properties=False, timeout=None, return_dict_and_exit=False):
    '''
    Given a dictionary of the form
    
    {'device1.property1.element1' : value1,
    'device2.property2.element2' : value2,}
    
    send a command to INDI and wait for the
    current state to match the target.
    
    This is just a light wrapper around
    INDIClient.wait_for_state.
    
    Parameters:
        cmd_dict : dict
            Dictionary of INDI commands
        tol : float or dict
            If a float, applies the same tolerance spec
            to all commands.
            If a dict, applies the tolerance for each entry and must have all the same keys
            as cmd_dict. This includes SWITCH type properties, even though the tolerance
            values are ignored for these.
    '''
    
    if wait_for_properties:
            properties = [k.rsplit('.', maxsplit=1)[0] for k in cmd_dict.keys()]
            client.wait_for_properties(properties, timeout=timeout) 


    toltype = type(tol)
    if toltype is dict and set(tol) != set(cmd_dict):
            missing = set(cmd_dict).difference(set(tol))
            raise ValueError('The tolerance dict must have all the same entries as the command dict!' \
                             f' Missing {list(missing)} keys in the tolerance dictionary.')
    
    # automate building the dictionary to pass to wait_for_state
    status_dict = {}
    for key, targ in cmd_dict.items():
        # figure out the associated property kind
        device, prop, element = key.split('.')
        kind = client.devices[device].properties[prop].KIND
        
        if toltype is dict:
            tolerance = tol[key]
        else:
            tolerance = tol

        if kind == indi.INDIPropertyKind.NUMBER:
            # for NUMBERs, set command 'target' and test 'current'
            current_dict = {
                'value' : targ,
                'test' : lambda current, value, tolerance=tolerance: abs(current - value) < tolerance
            }
            target_dict = {
                'value' : targ,
                'test' : lambda current, value, tolerance=tolerance: abs(current - value) < tolerance
            }
            status_dict[key] = target_dict
            status_dict[f'{device}.{prop}.current'] = current_dict
        elif kind == indi.INDIPropertyKind.SWITCH:
            # Set a SWITCH element and test the same element
            status_dict[key] = {'value' : targ}
        else:
            raise NotImplementedError('Only NUMBER and SWITCH INDI properties are supported!')
            
    if return_dict_and_exit:
        return status_dict
        
    return client.wait_for_state(status_dict, wait_for_properties=wait_for_properties, timeout=timeout)

def _is_open(func):
    def deco(self, *args, **kwargs):
        if self.is_open:
            return func(self, *args, **kwargs)
        else:
            raise RuntimeError('image stream is not open!')
    return deco

class ImageStream(shmio.Image):
    '''
    Convenience class to make interacting
    with image stream a little easier.
    '''
    SUCCESS_CODE = 0

    def __init__(self, name, expected_shape=None):
        super().__init__()
        self.name = name
        self.is_open = False
        self.open()

        if expected_shape is not None:
            self.check_shape(expected_shape)

        self.buffer = np.array(self, copy=False).T
        self.naxis = self.md.naxis
        self.semindex = None

    def __enter__(self):
        if not self.is_open:
            self.open(self.name)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.close()

    def __getitem__(self, start, stop, step):
        return np.array(self.buffer[start:stop:step], copy=True)

    def open(self):
        ret = super().open(self.name)
        if ret != self.SUCCESS_CODE:
            raise RuntimeError(f'Could not open shared memory image "{self.name}"!')
        else:
            self.is_open = True

    def check_shape(self, expected_shape):
        curshape = self.md.size
        if list(curshape) != list(expected_shape):
            logger.info(f'Got shape {curshape} but expected shape {expected_shape}. Destroying and re-creating.')
            self.destroy()
            self.create(self.name, expected_shape, shmio.ImageStreamIODataType.FLOAT, 1, 8)

    @_is_open
    def close(self):
        super().close()
        self.is_open = False

    @_is_open
    def grab_buffer(self):
        return np.array(self.buffer, copy=True)

    @_is_open
    def grab_latest(self):
        if self.naxis  < 3:
            return np.array(self.buffer, copy=True)
        else:
            cnt1 = self.md.cnt1
            #print(f'Got a semaphore! Buffer index: {self.md.cnt0}')
            return np.array(self.buffer[cnt1], copy=True)

    @_is_open
    def grab_many(self, n):
    	# find a free semaphore to wait on
        if self.semindex is None:
            self.semindex = self.getsemwaitindex(1)
            logger.info(f'Got semaphore index {self.semindex}.')
        i = 0
        cube = []
        # flush semaphores before collecting images
        self.semflush(self.semindex)
        while i < n:
        	# collect each new image
            self.semwait(self.semindex)
            cube.append(self.grab_latest())
            i += 1
        return cube

def create_shmim(name, dims, dtype=shmio.ImageStreamIODataType.FLOAT, shared=1, nbkw=8):
    # if ImageStream objects didn't auto-open on creation, you could create and return that instead. oops.
    img = shmio.Image()
    # not sure if I should try to destroy first in case it already exists
    img.create(name, dims, dtype, shared, nbkw)
    img.close()

def send_dm_poke(shmim_name, x, y, val):
    with ImageStream(shmim_name) as shmim:
        curvals = shmim.grab_latest()
        curvals[y, x] = val
        shmim.write(curvals)

def send_fits_to_shmim(shmim_name, fitsfile):
    from astropy.io import fits
    with ImageStream(shmim_name) as shmim:
        with fits.open(fitsfile) as f:
            data = f[0].data
        shmim.write(data.astype(shmim.buffer.dtype))

def send_shmim_to_fits(shmim_name, fitsfile, nimages=1):
    from astropy.io import fits
    with ImageStream(shmim_name) as shmim:
        if nimages == 1:
            data = shmim.grab_latest() #don't wait for a new image, just take the current
        else:
            data = np.squeeze(shmim.grab_many(nimages)) # grab the nimages newest images
    fits.writeto(fitsfile, data)

def send_zeros_to_shmim(shmim_name):
    with ImageStream(shmim_name) as shmim:
        zeros = np.zeros_like(shmim.buffer)
        shmim.write(zeros)

def console_send_dm_poke():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('shmim_name', type=str, help='Name of shared memory name (ex: dm00disp01)')
    parser.add_argument('x', type=int, help='x coordinate (0-indexed)')
    parser.add_argument('y', type=int, help='y coordinate (0-indexed)')
    parser.add_argument('val', type=float, help='poke size in microns')

    args = parser.parse_args()

    send_dm_poke(args.shmim_name, args.x, args.y, args.val)

def console_send_fits_to_shmim():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('shmim_name', type=str, help='Name of shared memory name (ex: dm00disp01)')
    parser.add_argument('fitsfile', type=str, help='Path to fits file.')
    args = parser.parse_args()

    send_fits_to_shmim(args.shmim_name, args.fitsfile)

def console_send_shmim_to_fits():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('shmim_name', type=str, help='Name of shared memory name (ex: dm00disp01)')
    parser.add_argument('fitsfile', type=str, help='Path to fits file.')
    parser.add_argument('--nimages', type=int, default=1, help='Number of images to grab')
    args = parser.parse_args()

    send_shmim_to_fits(args.shmim_name, args.fitsfile, args.nimages)

def console_send_zeros_to_shmim():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('shmim_name', type=str, help='Name of shared memory name to 0 out. (ex: dm00disp01)')
    args = parser.parse_args()

    send_zeros_to_shmim(args.shmim_name)

if __name__ == '__main__':
    pass
