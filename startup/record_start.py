import os
import time
from os import path

import libtmux


def get_next_output_dir(base_dir):
    """
    Find the smallest unused numbered directory in the base directory.

    Args:
        base_dir (str): The base directory to search in.

    Returns:
        str: The path to the next available directory.
    """
    existing_dirs = [
        int(d)
        for d in os.listdir(base_dir)
        if d.isdigit() and os.path.isdir(os.path.join(base_dir, d))
    ]
    next_dir = str(min(set(range(len(existing_dirs) + 1)) - set(existing_dirs)))
    return os.path.join(base_dir, next_dir)


if __name__ == "__main__":
    server = libtmux.Server(
        config_file=path.expandvars("/home/user/kinova_flow/startup/.tmux.conf")
    )
    if server.has_session("sim"):
        exit()
    else:
        session = server.new_session(
            "sim", start_directory="/home/user/kinova_flow", attach=False
        )

    # terminals for the simulation to start

    data_folder = "/home/user/kinova_flow/data/"
    output_dir = get_next_output_dir(data_folder)
    os.makedirs(output_dir, exist_ok=True)

    terminals = {
        "rqt": "rqt --perspective-file /home/user/kinova_flow/startup/rqt.perspective",
        "kortex_bringup": "roslaunch kortex_bringup kortex_bringup.launch",
        # "cameras": "roslaunch --wait cameras single_rs.launch",
        "joynode": 'rosparam set joy_node/dev "/dev/input/js0"\nrosrun joy joy_node',
        "kortex_joy": "rosrun joystick_control kinova_joy.py _controller:=cartesian",
        # "recorder": f"echo 'rosrun recorder recorder.py _output_dir:={output_dir}'",
    }

    for name, cmd in terminals.items():
        window = session.new_window(name, attach=False)
        window.select_layout(layout="tiled")
        pane = window.panes[0]
        time.sleep(0.1)
        pane.send_keys(cmd, suppress_history=True)
