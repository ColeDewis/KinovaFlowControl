#!/usr/bin/python3.10
import os
import pickle
import threading
from datetime import datetime

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from kortex_driver.msg import BaseCyclic_Feedback
from sensor_msgs.msg import Image, JointState


class Recorder:
    def __init__(self, output_dir, record_interval=0.1):
        """
        Initialize the Recorder class.

        Args:
            output_dir (str): Directory where the recorded data will be saved.
            record_interval (float): Interval in seconds for synchronized recording.
        """
        self.output_dir = output_dir
        self.record_interval = record_interval
        self.bridge = CvBridge()

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # TODO: need to modify somewhat to match what 3d diffusion policy
        # expects as the data format, then we can use their scripts.
        # It will be a bit different than here. 

        # Directories for saving data
        self.joint_states_dir = os.path.join(self.output_dir, "joint_states")
        self.cartesian_positions_dir = os.path.join(
            self.output_dir, "cartesian_positions"
        )
        self.rgb_image_dir = os.path.join(self.output_dir, "imgs")
        self.depth_image_dir = os.path.join(self.output_dir, "depth")

        os.makedirs(self.joint_states_dir, exist_ok=True)
        os.makedirs(self.cartesian_positions_dir, exist_ok=True)
        os.makedirs(self.rgb_image_dir, exist_ok=True)
        os.makedirs(self.depth_image_dir, exist_ok=True)

        # Data storage for synchronization
        self.joint_states = None
        self.cartesian_positions = None
        self.rgb_image = None
        self.depth_image = None
        self.lock = threading.Lock()

        # Shared index for sequential file naming
        self.shared_index = 0

        # Wait for topics to be available
        rospy.loginfo("Waiting for topics to become available...")
        rospy.wait_for_message("/my_gen3/joint_states", JointState)
        rospy.wait_for_message("/camera1/color/image_raw", Image)
        rospy.wait_for_message("/camera1/aligned_depth_to_color/image_raw", Image)
        rospy.wait_for_message("/my_gen3/base_feedback", BaseCyclic_Feedback)
        rospy.loginfo("All topics are available. Initializing subscribers...")

        # Subscribers
        self.joint_states_sub = rospy.Subscriber(
            "/my_gen3/joint_states", JointState, self.cache_joint_states
        )
        self.rgb_image_sub = rospy.Subscriber(
            "/camera1/color/image_raw", Image, self.cache_rgb_image
        )
        self.depth_image_sub = rospy.Subscriber(
            "/camera1/aligned_depth_to_color/image_raw", Image, self.cache_depth_image
        )
        self.cartesian_position_sub = rospy.Subscriber(
            "/my_gen3/base_feedback",
            BaseCyclic_Feedback,
            self.cache_cartesian_positions,
        )

        # Timer for synchronized recording
        self.timer = rospy.Timer(
            rospy.Duration(self.record_interval), self.record_synchronized_data
        )

        rospy.loginfo("Subscribers and timer initialized. Ready to start recording.")

    def save_array(self, array, directory, index):
        """
        Save a numpy array to a .npy file.

        Args:
            array (np.ndarray): The array to save.
            directory (str): The directory where the file will be saved.
            index (int): The index for the file name.
        """
        filepath = os.path.join(directory, f"{index}.npy")
        np.save(filepath, array)

    def save_image(self, image, directory, index):
        """
        Save an image to a .png file.

        Args:
            image (np.ndarray): The image to save.
            directory (str): The directory where the file will be saved.
            index (int): The index for the file name.
        """
        filepath = os.path.join(directory, f"{index}.png")
        cv2.imwrite(filepath, image)

    def cache_joint_states(self, msg):
        with self.lock:
            self.joint_states = {
                "timestamp": rospy.Time.now().to_sec(),
                "position": msg.position,
                "velocity": msg.velocity,
            }

    def cache_cartesian_positions(self, msg: BaseCyclic_Feedback):
        with self.lock:
            self.cartesian_positions = {
                "timestamp": rospy.Time.now().to_sec(),
                "position": [
                    msg.base.tool_pose_x,
                    msg.base.tool_pose_y,
                    msg.base.tool_pose_z,
                    msg.base.tool_pose_theta_x,
                    msg.base.tool_pose_theta_y,
                    msg.base.tool_pose_theta_z,
                ],
                "velocity": [
                    msg.base.tool_twist_linear_x,
                    msg.base.tool_twist_linear_y,
                    msg.base.tool_twist_linear_z,
                    msg.base.tool_twist_angular_x,
                    msg.base.tool_twist_angular_y,
                    msg.base.tool_twist_angular_z,
                ],
            }

    def cache_rgb_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            with self.lock:
                self.rgb_image = cv_image
        except Exception as e:
            rospy.logerr(f"Failed to cache RGB image: {e}")

    def cache_depth_image(self, msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            with self.lock:
                self.depth_image = depth_image
        except Exception as e:
            rospy.logerr(f"Failed to cache depth image: {e}")

    def record_synchronized_data(self, event):
        """
        Record all synchronized data at the specified interval.
        """
        with self.lock:
            rospy.loginfo(
                f"Recording synchronized data... {self.shared_index} {self.joint_states_dir}"
            )
            if self.joint_states:
                self.save_array(
                    self.joint_states, self.joint_states_dir, self.shared_index
                )
            if self.cartesian_positions:
                self.save_array(
                    self.cartesian_positions,
                    self.cartesian_positions_dir,
                    self.shared_index,
                )
            if self.rgb_image is not None:
                self.save_image(self.rgb_image, self.rgb_image_dir, self.shared_index)
            if self.depth_image is not None:
                self.save_array(
                    self.depth_image, self.depth_image_dir, self.shared_index
                )
            self.shared_index += 1

    def stop_recording(self):
        """
        Stop recording and clean up resources.
        """
        self.timer.shutdown()
        rospy.loginfo("Recording stopped.")


if __name__ == "__main__":
    try:
        # Get parameters from ROS parameter server
        rospy.init_node("recorder", anonymous=True)

        output_dir = rospy.get_param("~output_dir", "/tmp/recordings")
        record_interval = rospy.get_param("~record_interval", 0.1)

        recorder = Recorder(output_dir, record_interval)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
