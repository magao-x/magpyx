from purepyindi import INDIClient, SwitchState
from .utils import indi_send_and_wait

import numpy as np
from time import sleep

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('utils')

def get_indi_client(port=7624):
    client = INDIClient('localhost', port)
    client.start()
    return client

def set_camera_roi(client, device, roi_x, roi_y, roi_h, roi_w):
    # update roi parameters
    client.wait_for_properties([f'{device}.roi_region_x', f'{device}.roi_region_y', f'{device}.roi_region_h' ,f'{device}.roi_region_w',
                                f'{device}.roi_set'])
    client[f'{device}.roi_region_x.target'] = roi_x
    client[f'{device}.roi_region_y.target'] = roi_y
    client[f'{device}.roi_region_h.target'] = roi_h
    client[f'{device}.roi_region_w.target'] = roi_w
    sleep(2.0)
    client[f'{device}.roi_set.request'] = SwitchState.ON
    sleep(2.0)

def set_science_camera(client, camdevice, roi_dict=None, adcspeed=None, exptime=None, nddevice=None, nd=None, timeout=None):
    if roi_dict is not None:
        set_camera_roi(client, camdevice, roi_dict['roi_x'], roi_dict['roi_y'], roi_dict['roi_h'], roi_dict['roi_w'])
        logger.info(f'Set {camdevice} ROI.')
    if adcspeed is not None:
        client.wait_for_properties([f'{camdevice}.adcspeed',])
        client[f'{camdevice}.adcspeed.{adcspeed}'] = SwitchState.ON
        sleep(2.0)
        logger.info(f'Set {camdevice} adcspeed.')
    if exptime is not None:
        client.wait_for_properties([f'{camdevice}.exptime',])
        client[f'{camdevice}.exptime.target'] = exptime
        sleep(2.0)
        logger.info(f'Set {camdevice} exptime to {exptime}. Camera went to {client[f"{camdevice}.exptime.current"]}.')
    if (nddevice is not None) and (nd is not None):
        client.wait_for_properties([f'{nddevice}.filterName',])
        client[f'{nddevice}.filterName.{nd}'] = SwitchState.ON
        logger.info(f'Set {nddevice} to {nd}.')

def take_dark(camstream, client, camdevice, nimages, delay=2.0):
    client[f'{camdevice}.shutter.toggle'] = SwitchState.ON
    sleep(delay)
    dark = np.mean(camstream.grab_many(nimages),axis=0)
    client[f'{camdevice}.shutter.toggle'] = SwitchState.OFF
    return dark

def take_dark_fliptip(camstream, client, nimages):
    client[f'fliptip.position.out'] = SwitchState.ON
    sleep(1.0)
    dark = np.mean(camstream.grab_many(nimages),axis=0)
    client[f'fliptip.position.in'] = SwitchState.ON
    return dark

def move_stage(client, device, position, block=True, timeout=60):
    if block:
        command_dict = {f'{device}.position.target' : position}
        indi_send_and_wait(client, command_dict, tol=1e-2, wait_for_properties=True, timeout=timeout)
    else:
        client.wait_for_properties([f'{device}.position',],timeout=timeout)
        client[f'{device}.position.target'] = position
