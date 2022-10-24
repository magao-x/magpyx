import multiprocessing as mp
from time import time, sleep

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
        self._write = super().write
        self.name = name
        self.is_open = False
        self.open()

        if expected_shape is not None:
            self.check_shape(expected_shape)

        self.buffer = np.array(self, copy=False, order='F').T
        self.dtype = self.buffer.dtype
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

    def write(self, arr):
        return self._write(np.asfortranarray(arr.T, dtype=self.dtype))

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
            #self.create(self.name, expected_shape, shmio.ImageStreamIODataType.FLOAT, 1, 8)
            buffer = np.zeros(expected_shape)
            self.create(self.name, buffer, -1, True, 8, 1, shmio.ImageStreamIODataType.FLOAT, 1)

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
    def grab_many(self, n, cnt0_min=None):
        '''
        Grab the next n frames.

        If cnt0_min is set, wait until cnt0 reaches this value,
        and then start collecting frames.
        '''
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
            if (cnt0_min is not None) and (self.md.cnt0 < cnt0_min):
                continue
            cube.append(self.grab_latest())
            i += 1
        return cube

    @_is_open
    def grab_after(self, n, nwait):
        cnt0 = self.md.cnt0
        return self.grab_many(n, cnt0_min=cnt0+nwait)

class AsynchronousImageStream(mp.Process):
    '''
    Open an ImageStream on a separate process and return results in a queue. (threads get blocked by semwait)
    
    Example of usage:
    >>> stream = AsynchronousImageStream('camsci2') # open
    >>> nimages = 10
    >>> nwait = 2
    >>> stream.grab_asynchronous(stream.grab_after, (nimages, nwait)) # grab some images (call as many times as you want)
    >>> images = stream.get_queued_images() # get queued images
    >>> stream.stop() # close the thread, queues, and shmim
    '''
    
    def __init__(self, name):
        super().__init__()
        self.name = name

        # initialize queues
        self.queue_in = mp.Queue()
        self.queue_out = mp.Queue()

        self.start()
 
    def grab_asynchronous(self, func, args=None):
        task = [func, args, False]
        self.queue_in.put(task)
        
    def run(self):
        
        shmim = ImageStream(self.name)
        while True:
            task = self.queue_in.get() # get a new task
            funcstr, args, finish = task
            
            if finish:
                #self.queue_in.task_done()
                shmim.close()
                return True
            
            func = getattr(shmim, funcstr)

            try:
                if args is None:
                    self.queue_out.put_nowait(func())
                else:
                    self.queue_out.put_nowait(func(*args))
            except: # catch something here
                logger.info('Something went wrong. I dunno.')
    
    def get_queued_images(self):
        allims = []
        while (not self.queue_out.empty()):
            im = self.queue_out.get_nowait()
            allims.append(im)
        return allims
    
    def stop(self, wait=True, timeout=10):
        
        # stop the worker function (also closes the shmim)
        self.queue_in.put([None, None, True])
        
        # wait for the worker to stop
        t0 = time()
        while self.is_alive() and wait:
            sleep(0.1)
            t = time()
            if (t-t0) >= timeout:
                break
            
        # get abandoned images
        abandoned_images = self.get_queued_images()
 
        # close the queues
        self.queue_in.close()
        self.queue_out.close()
        
        # close the multiprocess worker
        mp.active_children()
        self.join(timeout=timeout)
        self.terminate()
        self.close()
        
        return abandoned_images

def create_shmim(name, dims, dtype=shmio.ImageStreamIODataType.FLOAT, shared=1, nbkw=8):
    # if ImageStream objects didn't auto-open on creation, you could create and return that instead. oops.
    img = shmio.Image()
    # not sure if I should try to destroy first in case it already exists
    #img.create(name, dims, dtype, shared, nbkw)
    buffer = np.zeros(dims)
    img.create(name, buffer, -1, True, 8, 1, dtype, 1)
    #img.close()

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
        zeros = np.zeros_like(shmim.buffer)#.T
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


def console_tweeter_um2V():
    from .dm import dmutils
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('um', type=float, help='Command in microns to convert to voltage send to tweeter DM')
    args = parser.parse_args()
    V = dmutils.tweeter_um_to_V(args.um)
    print(f'{V:.2f} Volts')

def console_tweeter_V2um():
    from .dm import dmutils
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('V', type=float, help='Command in V to convert into microns on tweeter DM')
    args = parser.parse_args()
    um = dmutils.tweeter_V_to_um(args.V)
    print(f'{um:.2f} um')

def str2bool(v):
    return str(v).lower() in ('true', '1')

if __name__ == '__main__':
    pass
