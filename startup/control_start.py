import os
import time
from os import path

import libtmux

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

    terminals = {
        "rqt": "rqt --perspective-file /home/user/kinova_flow/startup/rqt.perspective",
        "rviz": "rviz -d /home/user/kinova_flow/startup/conf.rviz",
        "kortex_bringup": "roslaunch kortex_bringup kortex_bringup.launch",
        "cameras": "roslaunch --wait cameras single_rs.launch",
        "joynode": 'rosparam set joy_node/dev "/dev/input/js0"\nrosrun joy joy_node',
        "kortex_joy": "rosrun joystick_control kinova_joy.py _controller:=cartesian",
    }

    for name, cmd in terminals.items():
        window = session.new_window(name, attach=False)
        window.select_layout(layout="tiled")
        pane = window.panes[0]
        time.sleep(0.1)
        pane.send_keys(cmd, suppress_history=True)
