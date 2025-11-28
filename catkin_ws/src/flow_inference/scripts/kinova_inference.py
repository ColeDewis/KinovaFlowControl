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
import cv_bridge

from kortex_driver.msg import BaseCyclic_Feedback

bridge = cv_bridge.CvBridge()

class FlowInference:
    def __init__(self):
        self.robot = KinovaGen3()

        rospy.on_shutdown(self.stop_robot)

        pc_subscriber = message_filters.Subscriber(
            "/camera1/depth/color/points", PointCloud2
        )
        joints_subscriber = message_filters.Subscriber(
            "/my_gen3/joint_states", JointState
        )
        image_subscriber = message_filters.Subscriber(
            "/camera1/color/image_raw", Image
        )

        self.pos_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.pos_callback)

        self.states = []
        self.pointclouds = []
        self.imgs = []
        self.obs_window_size = 2
        self.action_horizon = 8
        self.action_buffer = []


        # diff ckpts doesn't seem to make a difference here. either:
        # - A: not enough demos for the variety of bottle positions, or
        # - B: some inconsistency between training and inference data processing i'd guess
        # pretty sure its B; even with much easier bottle placements, its still doing things
        # that don't line up with observations. probably an inconsistency... 
        # will also try loading the model with the config explictly to see if that changes anything.
        self.model_workspace = TrainDP3Workspace.create_from_checkpoint(
            # "/home/user/kinova_flow/data/ckpts/bottleepoch=2600-test_mean_score=-0.010.ckpt"
            # "/home/user/kinova_flow/data/ckpts/kinova_pickup_bottle-epoch=0400-test_mean_score=-0.058.ckpt"
            # "/home/user/kinova_flow/data/ckpts/kinova_pickup_bottle-epoch=0800-test_mean_score=-0.043.ckpt"
            # "/home/user/kinova_flow/data/ckpts/kinova_pickup_bottle-epoch=1200-test_mean_score=-0.030.ckpt"
            # "/home/user/kinova_flow/data/ckpts/easy_kinova_pickup_bottle-epoch=0200-test_mean_score=-0.110.ckpt",
            # "/home/user/kinova_flow/data/ckpts/easy_kinova_pickup_bottle-epoch=0400-test_mean_score=-0.080.ckpt" # GOOD KINDA
            # "/home/user/kinova_flow/data/ckpts/easy_kinova_pickup_bottle-epoch=0600-test_mean_score=-0.063.ckpt"
            # "/home/user/kinova_flow/data/ckpts/imgs_easy_kinova_pickup_bottle-epoch=0350-test_mean_score=-0.069.ckpt"
            # "/home/user/kinova_flow/data/ckpts/imgs_easy_kinova_pickup_bottle-epoch=0200-test_mean_score=-0.091.ckpt"
            # "/home/user/kinova_flow/data/ckpts/velocity_easy_kinova_pickup_bottle-epoch=0400-test_mean_score=-0.061.ckpt"
            # "/home/user/kinova_flow/data/ckpts/position_easy_kinova_pickup_bottle-epoch=0400-test_mean_score=-0.016.ckpt"
            "/home/user/kinova_flow/data/ckpts/position_easy_kinova_pickup_bottle-epoch=1400-test_mean_score=-0.006.ckpt"
        )
        self.model_workspace.model.eval()
        self.model_workspace.model.cuda()

        # rospy.loginfo(f"{self.model_workspace.model.normalizer.get_input_stats()}")
        # rospy.loginfo(f"{self.model_workspace.model.normalizer.params_dict['agent_pos']['offset']}")
        # rospy.loginfo(f"{self.model_workspace.model.normalizer.get_input_stats()['point_cloud']['max']}")
        # rospy.loginfo(f"{self.model_workspace.model.normalizer.get_input_stats()['point_cloud']['min']}")
        # rospy.loginfo(f"{self.model_workspace.model.normalizer.get_output_stats()}")
        # exit()


        ts = message_filters.ApproximateTimeSynchronizer(
            [joints_subscriber, pc_subscriber], queue_size=1, slop=0.5
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
        self.last_pos = None
        self.control_loop()

    def pos_callback(self, feedback_msg):
        self.last_pos = np.array([
            feedback_msg.base.tool_pose_x,
            feedback_msg.base.tool_pose_y,
            feedback_msg.base.tool_pose_z,
            feedback_msg.base.tool_pose_theta_x,
            feedback_msg.base.tool_pose_theta_y,
            feedback_msg.base.tool_pose_theta_z,
        ])

    def infer_callback(self, joint_msg, pc_msg):
        numpy_pc = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
        xyz = ros_numpy.point_cloud2.get_xyz_points(numpy_pc, remove_nans=True)
        rgb_packed = numpy_pc['rgb'].view(np.uint32)
        r = (rgb_packed >> 16) & 255
        g = (rgb_packed >> 8) & 255
        b = rgb_packed & 255
        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
        rgb = rgb.reshape(-1, 3)
        points = np.concatenate([xyz, rgb], axis=-1)

        gripper_pos = 0.0 if joint_msg.position[8] < 0.1 else 1.0
        joints = np.concatenate([joint_msg.position[:7], [gripper_pos]])

        # img = bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
        # self.imgs.append(img)

        self.states.append(joints)
        self.pointclouds.append(preprocess_point_cloud(points))
        if len(self.states) > self.obs_window_size:
            self.states.pop(0)
            self.pointclouds.pop(0)
            # self.imgs.pop(0)
        elif len(self.states) < self.obs_window_size:
            return
        
        # empirically 2 seems to be needed in order for it to not lag behind
        if len(self.action_buffer) > 0:
            # don't do inference if we have actions to send still
            # NOTE 2 is kinda heuristic, but seems roughly correct
            return
        
        rospy.logwarn(f"{self.action_buffer}")
        state_input = np.array(self.states)
        pc_input = np.array(self.pointclouds)  # (obs_window_size, N, 6)
        imgs = np.array(self.imgs)

        self.model_workspace.model.eval()
        data = {
            "agent_pos": torch.from_numpy(state_input).unsqueeze(0).cuda(), 
            "point_cloud": torch.from_numpy(pc_input).unsqueeze(0).cuda(),
            # "img": torch.from_numpy(imgs).unsqueeze(0).cuda()
        }

        # rospy.loginfo(f"{data['agent_pos'].shape}, {data['point_cloud'].shape}")
        # rospy.loginfo(f'{data["point_cloud"][:, :, :, :3].shape} {data["point_cloud"][:, :, :, :3].min().item()}, {data["point_cloud"][:, :, :, :3].max().item()}')

        self.model_inference(data)
        
    def stop_robot(self):
        rospy.loginfo("SHUTTING DOWN: Stopping robot.")
        self.robot.send_twist_topic(np.zeros(6))

    def model_inference(self, data):
        with torch.no_grad():
            action_seq = self.model_workspace.model.predict_action(data)
            action = action_seq['action']
            actions = [a.cpu().numpy() for a in action.squeeze(0)]
            rounded_action = torch.round(action * 1000) / 1000
            rospy.loginfo(f"Predicted action sequence: {rounded_action}")

        # clear buffer so we never execute an old action sequence
        self.action_buffer = []
        self.action_buffer.extend(actions)
        self.action_buffer = [
            # NOTE: dropping first two actions seems to help smoothness a bit
            # self.action_buffer[0], 
            # self.action_buffer[1], 
            self.action_buffer[2],
            self.action_buffer[3],
            self.action_buffer[4],
            self.action_buffer[5],
            self.action_buffer[6],
            self.action_buffer[7],
        ]
    
    def send_safe_action(self, action):
        # TODO: also set certain vels to 0 if close to boundaries.
        action[:3] = np.clip(action[:3], -0.1, 0.1)  # xyz limits
        action[3:6] = np.deg2rad(np.clip(action[3:6], -5, 5))
        action[6] = np.clip(action[6], 0, 1)  #* 0.5 # gripper position, scale down to not crush stuff 
        action[6] = 0.0 if action[6] < 0.5 else 0.5
        rospy.loginfo(f"Sending action: {action}")
        self.robot.send_twist_topic(action[:6] * 0.5)
        self.robot.send_gripper_command(action[6])

    def send_pose_test(self, action):
        self.robot.send_pose(
            action[0],
            action[1],
            action[2],
            action[3],
            action[4],
            action[5],
        )
        action[6] = np.clip(action[6], 0, 1)  #* 0.5 # gripper position, scale down to not crush stuff 
        action[6] = 0.0 if action[6] < 0.5 else 0.5
        self.robot.send_gripper_command(action[6])

    def send_pose_velocity_test(self, action):
        if self.last_pos is None:
            rospy.loginfo("Haven't received position feedback yet, can't send velocity command.")
            return
        diff = action[:6] - self.last_pos

        diff[0:3] = diff[0:3] * 2
        diff[3:6] = np.deg2rad(diff[3:6]) * 0.5
        # diff[3] = diff[3] * -1
        diff[3:5] = 0.0  # gotta record angles properly for these to not be dumb
        # rospy.loginfo(f"\ndiff: {diff[3:6]}, \naction: {action[3:6]}, \nlast_pos: {self.last_pos[3:6]}")
        # rospy.loginfo(f"Sending velocity command: {diff}")
        self.robot.send_twist_topic(diff)  
        action[6] = np.clip(action[6], 0, 1)  
        action[6] = 0.0 if action[6] < 0.5 else 0.5
        self.robot.send_gripper_command(action[6])


    def control_loop(self):
        rate = rospy.Rate(15)  # 15 Hz, we SHOULD have recorded demos at that frequency
        # TODO might be better for demos to be delta positions rather than velocities tbh.

        while not rospy.is_shutdown():
            if len(self.action_buffer) == 0:
                rospy.loginfo("Waiting for next inference...")
                self.robot.send_twist_topic(np.zeros(6))
                rate.sleep()
                continue

            # grab next action, and send to robot.
            # rospy.loginfo(f"Next action from buffer: {next_action}")
            # self.send_pose_test(next_action)
            # next_action = self.action_buffer[0]
            next_action = self.action_buffer.pop(0)
            self.send_pose_velocity_test(next_action)

            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("flow_inference_node")
    flow_inference = FlowInference()
    rospy.spin()