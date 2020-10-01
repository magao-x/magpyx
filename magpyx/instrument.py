from time import sleep
from purepyindi import INDIClient, SwitchState
from magpyx.utils import indi_send_and_wait

def get_indi_client(port=7624):
    client = INDIClient('localhost', port)
    client.start()
    return client

def set_camera_roi(client, device, roi_x, roi_y, roi_h, roi_w):
    # update roi parameters
    client[f'{device}.roi_x.target'] = roi_x
    client[f'{device}.roi_y.target'] = roi_y
    client[f'{device}.roi_h.target'] = roi_h
    client[f'{device}.roi_w.target'] = roi_w
    sleep(2.0)
    client[f'{device}.roi_set.request'] = SwitchState.ON
    sleep(2.0)

def send_to_camera_preset(camstream, camdevice, roi_x, roi_y, roi_h, roi_w, stagedevice, focus_pos, exptime, ndpreset):
    camstream.close()
    # set roi
    set_camera_roi(client, camdevice, roi_x, roi_y, roi_h, roi_w)
    
    # set fwscind to pupil and move to pupil position
    indi_send_and_wait(client,
                  {f'{stagedevice}.position.target' : focus_pos,
                   f'fwscind.filterName.{ndpreset}' : SwitchState.ON})
    
    # set exposure time
    indi_send_and_wait(client, {f'{camdevice}.exptime.target' : exptime})
    sleep(1)
    
    #camstream = ImageStream(camdevice)
    camstream.open()

 def take_dark(camstream, client, camdevice, nimages):
    client[f'{camdevice}.shutter.toggle'] = SwitchState.ON
    sleep(0.5)
    dark = np.mean(camstream.grab_many(nimages),axis=0)
    client[f'{camdevice}.shutter.toggle'] = SwitchState.OFF
    return dark

def move_stage(client, device, position, block=True, timeout=60):
    if block:
        command_dict = {f'{device}.position.target' : position}
        indi_send_and_wait(client, command_dict, tol=1e-2, wait_for_properties=True, timeout=timeout)
    else:
        client.wait_for_properties(['{device}.position',],timeout=timeout))
        client[f'{device}.position.target'] = position