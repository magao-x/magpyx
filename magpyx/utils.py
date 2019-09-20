import purepyindi as indi

try:
    import ImageStreamIOWrap as shmio
except ImportError:
    print('Could not import ImageStreamIOWrap!')

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
        self.buffer = np.array(self, copy=False).T
        self.naxis = im.md.naxis

    def __getitem__(self, start, stop, step):
        return np.array(self.buffer[start:stop:step], copy=True)

    def grab_buffer(self):
        return np.array(self.buffer, copy=True)

    def grab_latest(self):
        if naxis  < 3:
            return np.array(self.buffer, copy=True)
        else:
            cnt1 = im.md.cnt1
            return np.array(self.buffer[:, :, cnt1], copy=True)

    def grab_many(self, n):
        i = 0
        cube = []
        while i < n:
            self.semwait(0) # wait on new image
            cube.append(self.grab_latest())
            i += 1
        return cube


if __name__ == '__main__':
    pass
