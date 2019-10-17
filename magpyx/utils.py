import purepyindi as indi
import numpy as np
import ImageStreamIOWrap as shmio
from time import sleep

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

class ImageStream(shmio.Image):
    '''
    Convenience class to make interacting with image stream circular
    buffers without copying the entire buffer every time a little
    easier.
    '''

    def __init__(self, name):
        super().__init__()
        self.open(name)
        if self.memsize == 0:
            raise RuntimeError(f'Could not open shared memory image "{name}"!')
        self.buffer = np.array(self, copy=False).T
        self.naxis = self.md.naxis
        self.semindex = None

    def __getitem__(self, start, stop, step):
        return np.array(self.buffer[start:stop:step], copy=True)

    def grab_buffer(self):
        return np.array(self.buffer, copy=True)

    def grab_latest(self):
        if self.naxis  < 3:
            return np.array(self.buffer, copy=True)
        else:
            cnt1 = self.md.cnt1
            #print(f'Got a semaphore! Buffer index: {self.md.cnt0}')
            return np.array(self.buffer[cnt1], copy=True)

    def grab_many(self, n):
    	# find a free semaphore to wait on
        if self.semindex is None:
            self.semindex = self.getsemwaitindex(0)
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

def send_dm_poke(shmim_name, x, y, val):
    shmim = ImageStream(shmim_name)
    curvals = shmim.grab_latest()
    curvals[y, x] = val
    shmim.write(curvals)

def console_send_dm_poke():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('shmim_name', type=str, help='Name of shared memory name (ex: dm00disp01)')
    parser.add_argument('x', type=int, help='x coordinate (0-indexed)')
    parser.add_argument('y', type=int, help='y coordinate (0-indexed)')
    parser.add_argument('val', type=float, help='poke size in microns')

    args = parser.parse_args()

    send_dm_poke(args.shmim_name, args.x, args.y, args.val)

if __name__ == '__main__':
    pass
