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


class FlowInference:
    def __init__(self):
        self.robot = KinovaGen3()

        pc_subscriber = message_filters.Subscriber(
            "/camera1/depth/color/points", PointCloud2
        )
        joints_subscriber = message_filters.Subscriber(
            "/my_gen3/joint_states", JointState
        )
        # self.image_subscriber = message_filters.Subscriber(
        #     "/camera1/color/image_raw", Image
        # )

        self.states = []
        self.pointclouds = []
        self.obs_window_size = 2
        self.action_horizon = 8

        self.model_workspace = TrainDP3Workspace.create_from_checkpoint("/home/user/kinova_flow/data/ckpts/epoch=2800-test_mean_score=-0.019.ckpt")
        self.model_workspace.model.cuda()


        ts = message_filters.ApproximateTimeSynchronizer(
            [joints_subscriber, pc_subscriber], queue_size=10, slop=0.5
        )
        ts.registerCallback(self.infer_callback)
        # for testing, create dummy input

        # dummy_state = np.array([np.random.rand(self.obs_window_size, 8).astype(np.float32)])
        # dummy_pc = np.array([np.random.rand(2, 1024, 6).astype(np.float32)])

        # dummy_data = {
        #     "agent_pos": torch.from_numpy(dummy_state).cuda(),
        #     "point_cloud": torch.from_numpy(dummy_pc).cuda()
        # }

        # self.model_inference(dummy_data)
        # exit()

    def infer_callback(self, joint_msg, pc_msg):
        numpy_pc = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        xyz = ros_numpy.point_cloud2.get_xyz_points(numpy_pc, remove_nans=True)
        rgb_packed = numpy_pc['rgb'].view(np.uint32)
        r = (rgb_packed >> 16) & 255
        g = (rgb_packed >> 8) & 255
        b = rgb_packed & 255
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        points = np.concatenate([xyz, rgb], axis=-1)


        gripper_pos = 0.0 if joint_msg.position[8] < 0.1 else 1.0
        joints = np.concatenate([joint_msg.position[:7], [gripper_pos]])

        self.states.append(joints)
        self.pointclouds.append(preprocess_point_cloud(points))
        if len(self.states) > self.obs_window_size:
            self.states.pop(0)
            self.pointclouds.pop(0)
        elif len(self.states) < self.obs_window_size:
            return
        
        state_input = np.array(self.states)
        pc_input = np.array(self.pointclouds)  # (obs_window_size, N, 6)

        self.model_workspace.model.eval()
        data = {
            "agent_pos": torch.from_numpy(state_input).unsqueeze(0).cuda(), 
            "point_cloud": torch.from_numpy(pc_input).unsqueeze(0).cuda()
        }

        rospy.loginfo(f"{data['agent_pos'].shape}, {data['point_cloud'].shape}")

        self.model_inference(data)
        

    def model_inference(self, data):
        with torch.no_grad():
            action_seq = self.model_workspace.model.predict_action(data)
            action = action_seq['action']

            rounded_action = torch.round(action * 100) / 100
            rospy.loginfo(f"Predicted action sequence: {rounded_action}")


if __name__ == "__main__":
    rospy.init_node("flow_inference_node")
    flow_inference = FlowInference()
    rospy.spin()