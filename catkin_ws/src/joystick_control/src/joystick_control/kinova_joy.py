#!/usr/bin/python3

import numpy as np
import rospy
from control_utils.ik_utils import (cartesian_control, joint_control,
                                    png_control, xbox_control)
from control_utils.kinova_gen3 import RGBDVision
from kortex_driver.msg import *
from kortex_driver.srv import *
from sensor_msgs.msg import Joy
from std_msgs.msg import Bool, Float32, Int32MultiArray


class CustomCommand:
    def __init__(self, ax, mode, trans_gain, rot_gain, wrist_gain):
        self.ax = ax
        self.mode = mode
        self.trans_gain = trans_gain
        self.rot_gain = rot_gain
        self.wrist_gain = wrist_gain


def gen_iris(base, xbox=False):
    class IrisRecord(base):
        def __init__(self, xbox=False):
            super(IrisRecord, self).__init__()
            self.mode = 0  # modes for control
            self.automatic = 0  # mode for whether it approaches automatically
            self.prev_button_2 = 0  # prev button 2 to prevent double clicks
            self.prev_gripper_cmd = 0.0  # prev gripper cmd
            self.gripper_cmd = 0.0  # gripper cmd
            self.axes_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # joystick cmd
            self.home_array = np.array(
                [0.1, 65, -179.9, -120, 0, 100, 90]
            )  # home array in deg
            self.xbox = xbox
            # self.home_array = np.rad2deg(
            #     np.array(
            #         [
            #             0.01426189238734954,
            #             0.7500500262488266,
            #             3.138938014615202,
            #             -2.20651178744055,
            #             0.004436293743853995,
            #             1.473773107225566,
            #             1.5707963267948966,
            #         ]
            #         # -1.6477009622253327] # TODO this gets ignored not sure why
            #     )
            # )
            self.home_array = np.rad2deg(
                np.array(
                    [
                        0.012327630165246581,
                        0.9606400220790603,
                        3.129544779134384,
                        -2.072680219496908,
                        -0.004924717487774011,
                        1.5311032423278106,
                        -1.5944846120589338,
                    ]
                )
            )
            self.send_joint_angles(self.home_array)  # sends robot home
            self.window_center = (424, 240)
            self.grip_center = None

            self.custom_commands = []
            print(f"Current mode = {self.mode}\a", end="\r")

        def init_joy_sub(self):
            # Init subscribers and publishers
            if self.xbox:
                self.joy_sub = rospy.Subscriber("/joy", Joy, self.xbox_callback)
            else:
                self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

        def init_tool_sub(self):
            self.tool_sub = rospy.Subscriber(
                "/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback
            )

        def mode_switch(self):
            self.mode = (self.mode + 1) % 2
            print(f"Current mode = {self.mode}\a", end="\r")
            return

        def grip_callback(self, msg):
            if self.grip_center is None:
                self.grip_center = msg.data

        def obj_callback(self, msg):
            self.obj_center = msg.data
            # print(self.obj_center)

        def slope_callback(self, msg):
            self.slope = msg.data

        def tool_callback(self, msg):
            self.tooldata = [
                msg.base.tool_pose_x,
                msg.base.tool_pose_y,
                msg.base.tool_pose_z,
                msg.base.tool_pose_theta_x,
                msg.base.tool_pose_theta_y,
                msg.base.tool_pose_theta_z,
            ]
            # print("TOOL DATA", self.tooldata)

        def xbox_callback(self, msg):
            MAXV_GR = 0.3
            self.buttons = msg.buttons
            self.axes_vector = msg.axes
            roll = msg.buttons[1] - msg.buttons[3]
            self.axes_vector = [msg.axes[1], msg.axes[0], (1/(msg.axes[5]+1.1) - 1/(msg.axes[2]+1.1))/10, -msg.axes[4]/2, msg.axes[3], roll]
            # self.axes_vector = [msg.axes[1], msg.axes[0], msg.axes[2], msg.axes[4], msg.axes[3], msg.axes[5]]

            if msg.buttons[0]:
                pass

            if msg.buttons[1]:
                pass

            if msg.buttons[2]:
                pass
                
            if msg.buttons[3]:
                pass

            if msg.buttons[4]: # LB - open gripper
                self.gripper_cmd = MAXV_GR

            elif msg.buttons[5]: # RB - close gripper
                self.gripper_cmd = -1 * MAXV_GR

            else: # both buttons 0 and 1 are zero
                self.gripper_cmd = 0.0

            if msg.buttons[6]:
                # Start button pressed, send robot home
                self.run = False
                self.send_joint_speeds_command(np.zeros(7))
                self.send_joint_angles(self.home_array)
                rospy.loginfo("Start button pressed: sending robot to starting position")
                self.run = True


            if msg.buttons[7]:
                pass
            
            if msg.buttons[8]:
                pass

            if msg.buttons[9]:
                pass

            if msg.buttons[10]:
                pass

        def joy_callback(self, msg):
            self.buttons = msg.buttons
            self.axes_vector = msg.axes
            MAXV_GR = 0.3

            # check for gripper commands
            if msg.buttons[0]:  # trigger button - close gripper
                self.gripper_cmd = -1 * MAXV_GR
            elif msg.buttons[1]:  # button by thumb - open gripper
                self.gripper_cmd = MAXV_GR
            else:  # both buttons 0 and 1 are zero
                self.gripper_cmd = 0.0

            if msg.buttons[2] == 0 and self.prev_button_2 == 1:
                self.mode_switch()

            self.prev_button_2 = msg.buttons[2]

            if msg.buttons[3]:
                # self.automatic = not self.automatic
                self.full_auto = not self.full_auto
                self.full_auto_toggle_pub.publish(Bool(self.full_auto))

            if msg.buttons[4]:
                pass

            if msg.buttons[5]:
                pass

            if msg.buttons[6]:
                pass

            if msg.buttons[7]:
                pass

            if msg.buttons[8]:
                # button 8 pressed, send robot home
                self.send_joint_angles(self.home_array)
                rospy.loginfo("Button 8 pressed: sending robot to starting position")

            if msg.buttons[9]:
                pass

            if msg.buttons[10]:
                pass

            if msg.buttons[11]:
                pass

        def orient(self):
            # reset the custom commands
            center, trajectory = self.center(self.obj_center, self.window_center)

            cont = False

            # initial angeling of the robot
            if self.tooldata[3] < 105:
                axes_vector = np.zeros_like(self.axes_vector)
                axes_vector[1] = 1
                mode = 1
                self.custom_commands.append(
                    CustomCommand(axes_vector, mode, 0.2, 0.2, 0.2)
                )

            # move the object to the center of the robot vision
            elif not center:
                print("CENTER", abs(self.grip_center[0] - self.obj_center[0]))
                axes_vector = np.zeros_like(self.axes_vector)
                mode = 0
                if abs(self.grip_center[0] - self.obj_center[0]) > 20:
                    axes_vector[0] = (
                        np.sign(self.grip_center[0] - self.obj_center[0]) * 0.5
                    )
                axes_vector[4] = trajectory[0] / 2
                # axes_vector[3] = -1
                if self.tooldata[2] > 0.15 and self.tooldata[2] < 0.3:
                    axes_vector[5] = trajectory[1]
                else:
                    cont = True
                self.custom_commands.append(
                    CustomCommand(axes_vector, mode, 0.1, 0.2, 0.2)
                )

            # align the robot with the object
            if center or cont:
                if abs(self.slope) > 50:
                    axes_vector = np.zeros_like(self.axes_vector)
                    mode = 1
                    axes_vector[0] = np.sign(self.slope) * 0.5
                    self.custom_commands.append(
                        CustomCommand(axes_vector, mode, 0.2, 0.5, 0.2)
                    )

        def center(self, point1, point2):
            diff_x = point1[0] - point2[0]
            diff_y = point1[1] - point2[1]

            thresh_x = 10
            thresh_y = 30

            if abs(diff_x) < thresh_x and abs(diff_y) < thresh_y:
                return True, 0
            else:
                if abs(diff_x) < thresh_x:
                    x_traj = 0
                else:
                    x_traj = -1 if diff_x > thresh_x else 1
                if abs(diff_y) < thresh_y:
                    y_traj = 0
                else:
                    y_traj = -1 if diff_y > thresh_y else 1
                return False, [x_traj, y_traj]

        def step(self):
            # step according to rospy rate
            super().step(self.axes_vector, self.mode)
            if self.gripper_cmd != self.prev_gripper_cmd:
                success = self.send_gripper_command(self.gripper_cmd)
                self.prev_gripper_cmd = self.gripper_cmd

    return IrisRecord(xbox)


def main():
    xbox = rospy.get_param("~xbox", True)

    controller_name = rospy.get_param("~controller", "png_control")
    if controller_name == "png_control":
        controller = png_control  # can replace with cartesian_control or joint_control
    elif controller_name == "cartesian":
        if xbox:
            controller = xbox_control
        else:
            controller = cartesian_control
    else:
        rospy.logerr(f"Unknown controller: {controller_name}")

    rospy.loginfo(f"Starting controller: {controller_name}, xbox: {xbox}")
    robot = gen_iris(controller, xbox)
    robot.init_joy_sub()
    robot.init_tool_sub()

    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        robot.step()
        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.sleep(3)
        rospy.init_node("iris_control", anonymous=True)
        main()
    except rospy.ROSInterruptException:
        print("ROSInterruptException")
