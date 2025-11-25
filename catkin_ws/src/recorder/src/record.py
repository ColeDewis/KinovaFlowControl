#!/usr/bin/python3

import curses
import os
import select
import sys
import termios
import threading
import time
import tty

import cv2
import message_filters
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
import tf2_ros
from cv_bridge import CvBridge
from flow_inference.HumanScoredFlowMatching.scripts.convert_real_robot_data import \
    preprocess_point_cloud
from kortex_driver.msg._BaseCyclic_Feedback import BaseCyclic_Feedback
from sensor_msgs.msg import Image, JointState, Joy, PointCloud2
from std_msgs.msg import Float64, Int16
from tqdm import tqdm

bridge = CvBridge()

class Recorder:
    def __init__(self, dataset):
        rospy.init_node('record_py')

        # data folder setup stuff
        self.base_path = os.path.join(os.getcwd(), "data", dataset)
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # episode will be appended if we already started recording
        existing = [
            int(name) for name in os.listdir(self.base_path)
            if os.path.isdir(os.path.join(self.base_path, name)) and name.isdigit()
        ]
        self.episode_num = max(existing) + 1 if existing else 0

        tool_sub = message_filters.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback)
        joints_sub = message_filters.Subscriber('/my_gen3/joint_states', JointState)
        image_sub = message_filters.Subscriber('/camera1/color/image_raw', Image)
        pointcloud_sub = message_filters.Subscriber('/camera1/depth/color/points', PointCloud2)

        ts = message_filters.ApproximateTimeSynchronizer([joints_sub, tool_sub, pointcloud_sub, image_sub], 100, slop=0.5)
        ts.registerCallback(self.syncCallback)

        # desyncing check setup
        self.last_sync_time = rospy.Time.now()
        self.sync_timeout = rospy.Duration(1.0)
        rospy.Timer(rospy.Duration(0.5), self.checkDesync)

        # keyboard setup
        # self.exit_requested = False

        # self.old_attrs = termios.tcgetattr(sys.stdin)
        # tty.setcbreak(sys.stdin.fileno())
        # rospy.on_shutdown(self.resetKeyboard)

        # self.key_thread = threading.Thread(target=self.keyboardListener)
        # self.key_thread.daemon = True
        # self.key_thread.start()

        self.last_press = 'end'
        self.time_last = time.time()
    
    def __del__(self):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attrs)

    def init_joy_sub(self):
        # Init subscribers and publishers
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

    def syncCallback(self, joint_msg, cart_msg, pointcloud_msg, img_msg):
        # it is possible something breaks and we become desynced, dont want to record episode in this case
        self.last_sync_time = rospy.Time.now()

        # havent started yet
        if not hasattr(self, "curr_low_dim"):
            return
       
        try:
            # numpy_pc = np.column_stack([ros_numpy.numpify(depth)[f] for f in ("x", "y", "z", "rgb")])
            cv_image = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            self.curr_rgb.append(cv_image)
            self.curr_depth.append(pointcloud_msg)

            joint_msg.position[8] = 0.0 if joint_msg.position[8] < 0.5 else 1.0
            data = {
                # clip joints to only the main 7DOF
                # I believe gripper is 8, should verify
                'joints': {
                    'position': joint_msg.position[:8],
                    'velocity': joint_msg.velocity[:7],
                },
                'cartesian': {
                    "position": [
                        cart_msg.base.tool_pose_x,
                        cart_msg.base.tool_pose_y,
                        cart_msg.base.tool_pose_z,
                        cart_msg.base.tool_pose_theta_x,
                        cart_msg.base.tool_pose_theta_y,
                        cart_msg.base.tool_pose_theta_z,
                    ],
                    "velocity": [
                        cart_msg.base.tool_twist_linear_x,
                        cart_msg.base.tool_twist_linear_y,
                        cart_msg.base.tool_twist_linear_z,
                        cart_msg.base.tool_twist_angular_x,
                        cart_msg.base.tool_twist_angular_y,
                        cart_msg.base.tool_twist_angular_z,
                    ],
                },
            }
            
            self.curr_low_dim.append(data)
        except Exception as e:
            rospy.logerr("error in sync: %s", e)
   
    def checkDesync(self, event):
        now = rospy.Time.now()
        time_since_last_sync = now - self.last_sync_time

        if time_since_last_sync > self.sync_timeout:
            rospy.logwarn("Desynced, cancelling episode. Press space to rerun or q to exit")
            self._call_clear_faults()

    def joy_callback(self, msg):
        self.buttons = msg.buttons

        if msg.buttons[7]:
            if self.last_press == 'start' and time.time() - self.time_last > 0.5:
                rospy.loginfo("Episode ended")
                self.handleEpisodeEnd()
                self.last_press = 'end'
                self.time_last = time.time()
            elif self.last_press == 'end' and time.time() - self.time_last > 0.5:
                rospy.loginfo("Episode started")
                self.handleEpisodeStart()
                self.last_press = 'start'
                self.time_last = time.time()

    # dont echo key and allow non blocking
    # def keyboardListener(self):
    #     self.old_attrs = termios.tcgetattr(sys.stdin)
    #     tty.setcbreak(sys.stdin.fileno())

    #     rospy.loginfo("\n\n--------------------------------------------------\n\nPress q to quit, space to start or retry the episode. n to finish the episode early and go onto the next. Press h to send home")
    #     rospy.loginfo("\n\n Press space to begin.")

    #     try:
    #         while not rospy.is_shutdown():
    #             if self.exit_requested:
    #                 break

    #             if select.select([sys.stdin], [], [], 0.1)[0]:
    #                 ch1 = sys.stdin.read(1)
    #                 if ch1 == '\x1b':
    #                     # Possibly an arrow key, read the next two characters
    #                     ch2 = sys.stdin.read(1)
    #                     ch3 = sys.stdin.read(1)
    #                     key = ch1 + ch2 + ch3
    #                 else:
    #                     key = ch1
    #                 self.handleKeypress(key)
    #     finally:
    #         self.resetKeyboard()

    # def resetKeyboard(self):
    #     termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_attrs)

    # def handleKeypress(self, key):
    #     if key == 'q':
    #         rospy.loginfo("Pressed 'q': shutting down.")
    #         self.exit_requested = True
    #         rospy.signal_shutdown("Quit key pressed.")
    #     elif key == ' ':
    #         rospy.loginfo("Pressed space: start episode.")
    #         self.handleEpisodeStart()
    #     elif key == 'n':
    #         rospy.loginfo("Pressed n: end episode.")
    #         self.handleEpisodeEnd()
    #     # elif key == 'h':
    #     #     # go home
    #     #     self.send_joint_speeds_command(np.zeros(7))
    #     #     self.send_joint_angles(self.home_array)
    #     else:
    #         rospy.loginfo(f"Pressed '{key}' — no action assigned.")

    def handleEpisodeStart(self):
        rospy.loginfo("Starting episode " + str(self.episode_num))
        self.last_sync_time = rospy.Time.now()
        self.curr_low_dim = []
        self.curr_depth = []
        self.curr_rgb = []
   
    def handleEpisodeEnd(self):
        rospy.loginfo(f"Episode {self.episode_num} recorded")

        # create episode folder
        current_folder = os.path.join(self.base_path, str(self.episode_num))
        os.makedirs(current_folder)

        for i, idx in enumerate(tqdm([i for i in range(len(self.curr_low_dim))], desc="Saving frames")):
            # save each frame
            frame_folder = os.path.join(current_folder, str(i))
            os.makedirs(frame_folder)
            np.save(os.path.join(frame_folder, "low_dim.npy"), self.curr_low_dim[idx])
            np.save(os.path.join(frame_folder, "rgb.npy"), self.curr_rgb[idx])
            # TODO calibrate extrinsics...
            np.save(os.path.join(frame_folder, "depth.npy"), preprocess_point_cloud(self.curr_depth[idx]))

        rospy.loginfo(f"Finished saving episode {self.episode_num}")
        self.episode_num += 1

    # IF we need to subsample we can use this function. Otherwise don't use.
    # indexes = self.select_evenly_spaced([i for i in range(len(self.curr_low_dim))], max_length=512)
    def select_evenly_spaced(self, array, max_length=48):
        n = len(array)
        if n <= max_length:
            return array
        indices = np.linspace(0, n - 1, max_length, dtype=int)
        return [array[i] for i in indices]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python record.py <dataset_name>")
        sys.exit(1)

    dataset = sys.argv[1]

    recorder = Recorder(dataset)
    recorder.init_joy_sub()

    rospy.spin()