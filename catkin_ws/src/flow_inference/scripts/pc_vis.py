#!/usr/bin/env python3
import message_filters
import numpy as np
import ros_numpy
import rospy
import torch
from flow_inference.HumanScoredFlowMatching.flow_policy.train import \
    TrainDP3Workspace
from flow_inference.HumanScoredFlowMatching.scripts.convert_real_robot_data import \
    preprocess_point_cloud
from kortex_bringup import KinovaGen3
from sensor_msgs.msg import Image, JointState, PointCloud2

from visualizer import Visualizer, visualize_pointcloud

class PCVis:
    def __init__(self):

        self.vis = Visualizer()
        
    #     self.pc_subscriber = rospy.Subscriber(
    #         "/camera1/depth/color/points", PointCloud2, self.pc_callback
    #     )

    # def pc_callback(self, pc_msg):
        pc_msg = rospy.wait_for_message("/camera1/depth/color/points", PointCloud2)
        rospy.loginfo("got pc")
        numpy_pc = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        xyz = ros_numpy.point_cloud2.get_xyz_points(numpy_pc, remove_nans=True)
        rgb_packed = numpy_pc['rgb'].view(np.uint32)
        r = (rgb_packed >> 16) & 255
        g = (rgb_packed >> 8) & 255
        b = rgb_packed & 255
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        points = np.concatenate([xyz, rgb], axis=-1)
        points = preprocess_point_cloud(points)
        try:
            self.vis.visualize_pointcloud(points)
        except Exception as e:
            return



if __name__ == "__main__":
    rospy.init_node("pc_vis", anonymous=False)
    pc_vis = PCVis()
    # rospy.spin()