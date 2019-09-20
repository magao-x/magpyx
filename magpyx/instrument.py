import purepyindi as indi
from purepyindi.constants import parse_string_into_enum

from configparser import ConfigParser
import curses

from .utils import indi_send_and_wait
from time import sleep
import os

class CaseConfigParser(ConfigParser):
    '''Handle case and INDI enums'''
    
    def optionxform(self, optionstr):
        return optionstr
    
    def to_dict(self):
        return {k : dict({ks : parse_value(vs) for (ks, vs) in v.items()}) 
                for (k, v) in self._sections.items()}
    
    def get(self, section, option):
        try:
            return parse_value(ConfigParser.get(self, section, option))
        except ConfigParser.NoOptionError:
            return None

def parse_presets(configfile):
    if not os.path.isfile(configfile):
        raise FileNotFoundError(f'Could not find the preset file at {configfile}!')
    config = CaseConfigParser()
    config.read(configfile)
    return config.to_dict()

def print_presets_full(preset_dict):
    for k, v in preset_dict.items():
        print(f'{k}')
        for ks, vs in v.items():
            print(f'\t{ks}: {vs}')
            
def print_presets(preset_dict):
    for k in preset_dict.keys():
        print(f'{k}')
        
def parse_value(value):
    '''
    Attempt to parse into a double
    or other INDI Property.
    
    If all attempts fail, assume it's just a
    plain old string.
    '''
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return parse_string_into_enum(value, indi.SwitchState)
    except ValueError:
        pass
    try:
        return parse_string_into_enum(value, indi.PropertyState)
    except ValueError:
        pass
    return value


def console_status(status_dict, presetname='', auto_exit=True):
    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    
    stdscr.clear()

    # status colors
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    
    try:
        exit = False
        while not exit:
            report_state(status_dict, presetname, stdscr)
            sleep(0.5)
            if auto_exit:
                exit = all([v['status'] == 'READY' for v in status_dict.values()])
        report_state(status_dict, presetname, stdscr)
        sleep(5)
    except KeyboardInterrupt:
        curses.echo()
        curses.nocbreak()
        curses.endwin()
    finally:
        curses.echo()
        curses.nocbreak()
        curses.endwin()

        
def report_state(status_dict, presetname, stdscr):

    num_rows, num_cols = stdscr.getmaxyx()

    # title line
    topstr = f'Moving to configuration {presetname}!'
    cen = get_center_pos(num_cols, topstr)
    stdscr.addstr(0, cen, topstr, curses.A_BOLD)
    stdscr.clrtoeol()
        
    for i, (k, v) in enumerate(status_dict.items()):
        print_device_status(stdscr, k, v, i*3+1)

    stdscr.refresh()
    
def get_center_pos(ncols, message):
    # Calculate center column, and then adjust starting position based
    # on the length of the message
    half_length_of_message = int(len(message) / 2)
    middle_column = int(ncols / 2)
    x_position = middle_column - half_length_of_message
    return x_position
    
def get_column_pos(sec, ncols, nsec):
    seccols = int(ncols / nsec)
    return sec*seccols

def print_device_status(stdscr, name, devdict, row):
    
    num_rows, num_cols = stdscr.getmaxyx()
    
    # device name
    stdscr.addstr(row, 0, f'{name}', curses.A_UNDERLINE)
    stdscr.clrtoeol()
    
    # requested
    col1 = get_column_pos(0, num_cols, 4)
    requested = devdict['requested']
    stdscr.addstr(row+1, col1, f'Requested: {requested}')
    stdscr.clrtoeol()

    # current
    col2 = get_column_pos(1, num_cols, 4)
    current = devdict['current']
    stdscr.addstr(row+1, col2, f'Current: {current}')
    stdscr.clrtoeol()
    
    # target
    col3 = get_column_pos(2, num_cols, 4)
    target = devdict['target']
    stdscr.addstr(row+1, col3, f'Target: {target}')
    stdscr.clrtoeol()
    
    # status
    status = devdict['status']
    col4 = get_column_pos(3, num_cols, 4)
    stdscr.addstr(row+1, col4, f'Status: ')
    if status == 'READY':
        stdscr.addstr(f'{status}', curses.color_pair(1) | curses.A_BOLD)
    elif status == 'MOVING':
        stdscr.addstr(f'{status}', curses.color_pair(3) | curses.A_BOLD)
    elif status == 'TARGET CHANGED':
        stdscr.addstr(f'{status}', curses.color_pair(2) | curses.A_BOLD | curses.A_BLINK)
    stdscr.clrtoeol()

def indi_send_status(client, cmd_dict, presetname, tol=1e-3, status_dict=None, wait_for_properties=True, curses=False, curses_auto_exit=True, timeout=None):
    '''
    Send a list of commands
    and return a dictionary
    with the status of each
    property/element.
    
    status_dict has keys:
    
             'current' : None,
            'target' : None,
            'status' : 'UNKNOWN',
            'requested' : v,
            'watcher'
    
    ''' 
    # initialize status_dict
    if status_dict is None:
        status_dict = {key : {} for key in cmd_dict.keys()}

    # initialize the test dict from the cmd dict
    test_dict = indi_send_and_wait(client, cmd_dict, tol=tol,
                                   wait_for_properties=wait_for_properties,
                                   return_dict_and_exit=True,
                                   timeout=timeout)
    def watcher_closure(prop):
        
        for elem in prop.elements.values():
            key = elem.identifier
            if key not in status_dict.keys():
                continue
            curdict = status_dict[key]
            requested = test_dict[key]['value']

            if curdict['current_element'] is not None:
                current = curdict['current_element'].value
            else:
                current = elem.value
            if curdict['target_element'] is not None:
                target = curdict['target_element'].value
            else:
                target = elem.value
                 
            if 'test' in test_dict[key]:
                ready = test_dict[key]['test'](current, requested)
            else:
                ready = requested == current

            if ready:
                status = 'READY'
            else:
                status = 'MOVING'

            if 'test' in test_dict[key] and not test_dict[key]['test'](target, requested):
                status = 'TARGET CHANGED'

            status_dict[key].update({
                'status' : status,
                'current' : current,
                'target' : target,
            })
        
    for key, value in cmd_dict.items():
        # register the watchers
        devname, propname, elemname = key.split('.')
        
        prop = client.devices[devname].properties[propname]
        elem = prop.elements[elemname]

        # add/update an entry in the status dict
        if f'{devname}.{propname}.current' in test_dict.keys():
            current_element = client.devices[devname].properties[propname].elements['current']
        else:
            current_element = None

        if f'{devname}.{propname}.target' in test_dict.keys():
            target_element = client.devices[devname].properties[propname].elements['target']
        else:
            target_element = None

        cur_dict = {
            'status' : 'UNKNOWN',
            'target' : None,
            'requested' : value,
            'watcher' : watcher_closure,
            'current': None,
            'current_element' : current_element,
            'target_element' : target_element
        }
        
        status_dict[key].update(cur_dict)
        
        prop.add_watcher(watcher_closure)
        # force an initial update 
        watcher_closure(prop)
    
    # send the actual commands
    for key, value in cmd_dict.items():
        client[key] = value
        
    if curses:
        console_status(status_dict, presetname, auto_exit=curses_auto_exit)
    else:
        while not all([v['status'] == 'READY' for v in status_dict.values()]):
            sleep(0.1)
        print('All devices in position!')

    return status_dict

def send_indi_triplet():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('triplet', type=str, nargs='+', help='Move to a instrument preset')
    parser.add_argument('--host', type=str, default='localhost', help='INDI host [Default: localhost')
    parser.add_argument('--port', type=int, default=7624, help='INDI host [Default: localhost')
    parser.add_argument('--nocurses', action='store_true', help='Disable curses status reporting.')
    parser.add_argument('--hold', action='store_true', help='Keep curses status up until sigint is received [Default: exit 5 sec after all devices are ready].')
    parser.add_argument('--timeout', type=float, default=10., help='Time out after trying for X seconds [Default: 10 sec].')

    args = parser.parse_args()

    triplet_list = args.triplet

    cmd_dict = {}
    for triplet in triplet_list:
        k, v = triplet.split('=')
        cmd_dict[k] = parse_value(v)

    client = indi.INDIClient(args.host, args.port)
    client.start()

    if not args.nocurses:
        indi_send_status(client, cmd_dict, '[CUSTOM]', curses_auto_exit=not args.hold, curses=True, timeout=args.timeout)
    else:
        indi_send_and_wait(client, cmd_dict, wait_for_properties=True, timeout=args.timeout)

def main():
    # parse command line arguments
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('preset', type=str, nargs='?', default=None, help='Move to a instrument preset')
    group.add_argument('--ls',  action='store_true', help='Print all instrument presets, as defined in /opt/MagAOX/config/inspresets.conf.')
    group.add_argument('--lsp', type=str, default=None, help='Print all INDI properties associated with a given preset.')
    group.add_argument('--lsa',  action='store_true', help='Print all INDI properties for all presets.')
    parser.add_argument('--host', type=str, default='localhost', help='INDI host [Default: localhost')
    parser.add_argument('--port', type=int, default=7624, help='INDI host [Default: localhost')
    parser.add_argument('--nocurses', action='store_true', help='Disable curses status reporting.')
    parser.add_argument('--hold', action='store_true', help='Keep curses status up until sigint is received [Default: exit 5 sec after all devices are ready].')
    parser.add_argument('--timeout', type=float, default=10., help='Time out after trying for X seconds [Default: 10 sec].')


    args = parser.parse_args()

    if len(sys.argv[1:])==0:
        parser.print_help()
        return

    presets = parse_presets('/opt/MagAOX/config/inspresets.conf')

    if args.preset is not None:
        client = indi.INDIClient(args.host, args.port)
        client.start()
        if not args.nocurses:
            indi_send_status(client, presets[args.preset], args.preset, curses_auto_exit=not args.hold, curses=True, timeout=args.timeout)
        else:
            indi_send_and_wait(client, presets[args.preset], wait_for_properties=True, timeout=args.timeout)

    if args.ls:
        print_presets(presets)
    elif args.lsp is not None:
        preset = presets.get(args.lsp, None)
        if preset is not None:
            print_presets(preset)
        else:
            print(f'Could not find requested preset. Valid options are {list(presets.keys())}.')
    elif args.lsa:
        print_presets_full(presets)

if __name__ == '__main__':
    main()

