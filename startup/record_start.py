import os
import time
from os import path

import libtmux


def get_next_output_dir(base_folder, prefix="dataset-"):
    existing_dirs = [
        d
        for d in os.listdir(base_folder)
        if path.isdir(path.join(base_folder, d)) and d.startswith(prefix) and d[len(prefix):].isdigit()
    ]
    existing_indices = [int(d[len(prefix):]) for d in existing_dirs]
    next_index = max(existing_indices) + 1 if existing_indices else 1
    return path.join(base_folder, f"{prefix}{next_index}")

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

    dataset_folder = "/home/user/kinova_flow/data/"
    if not path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    dataset_name = get_next_output_dir(dataset_folder)

    terminals = {
        "rqt": "rqt --perspective-file /home/user/kinova_flow/startup/rqt.perspective",
        "rviz": "rviz -d /home/user/kinova_flow/startup/conf.rviz",
        "kortex_bringup": "roslaunch kortex_bringup kortex_bringup.launch",
        "cameras": "roslaunch --wait cameras single_rs.launch",
        "joynode": 'rosparam set joy_node/dev "/dev/input/js0"\nrosrun joy joy_node',
        "kortex_joy": "rosrun joystick_control kinova_joy.py _controller:=cartesian",
        "recorder": f"rosrun recorder record.py {dataset_name}",
    }

    for name, cmd in terminals.items():
        window = session.new_window(name, attach=False)
        window.select_layout(layout="tiled")
        pane = window.panes[0]
        time.sleep(0.1)
        pane.send_keys(cmd, suppress_history=True)
