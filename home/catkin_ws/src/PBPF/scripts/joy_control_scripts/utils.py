from key_mapper import KEYMAP

def print_keys_pressed(buttons_list):
    '''
    Given a list representing the state of the keys of a PS3 controller it
    will print which key was pressed.
    '''
    for key, num in KEYMAP.items():
        if buttons_list[num]:
            print(f'Key {key} pressed!')

