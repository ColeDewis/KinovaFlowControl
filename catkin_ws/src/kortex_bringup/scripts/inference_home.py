#!/usr/bin/python3

import numpy as np
import rospy
from kortex_bringup import KinovaGen3

if __name__ == "__main__":
    # Init ros node
    rospy.init_node("send_gen3_home", anonymous=False)

    # Robot node
    gen3 = KinovaGen3()

    print()
    # no realsense mounted:
    # angles = np.deg2rad(
    #     np.array(
    #         [
    #             -0.1336059570312672,
    #             -28.57940673828129,
    #             -179.4915313720703,
    #             -147.7,
    #             0.06742369383573531,
    #             -57.420898437500036,
    #             89.88030242919922,
    #         ]
    #     )
    # )
    angles = np.array(
        [0.16499872527316523, 0.5617058144781881, -3.121019201536713, -2.236882210195705, 0.1763588217323049, 1.2173782053845685, -1.674093953892454]
    )
    success = gen3.send_joint_angles(angles)
    print("Done. Joints:", gen3.position)
