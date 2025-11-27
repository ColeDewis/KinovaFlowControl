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

        rospy.on_shutdown(self.stop_robot)

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
        self.action_buffer = []

        self.model_workspace = TrainDP3Workspace.create_from_checkpoint("/home/user/kinova_flow/data/ckpts/bottleepoch=2600-test_mean_score=-0.010.ckpt")
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
        
        self.control_loop()

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
        
        # empirically 2 seems to be needed in order for it to not lag behind
        if len(self.action_buffer) > 2:
            # don't do inference if we have actions to send still
            # NOTE 2 is kinda heuristic, but seems roughly correct
            return
        
        state_input = np.array(self.states)
        pc_input = np.array(self.pointclouds)  # (obs_window_size, N, 6)

        self.model_workspace.model.eval()
        data = {
            "agent_pos": torch.from_numpy(state_input).unsqueeze(0).cuda(), 
            "point_cloud": torch.from_numpy(pc_input).unsqueeze(0).cuda()
        }

        # rospy.loginfo(f"{data['agent_pos'].shape}, {data['point_cloud'].shape}")

        self.model_inference(data)
        
    def stop_robot(self):
        rospy.loginfo("SHUTTING DOWN: Stopping robot.")
        self.robot.send_twist_topic(np.zeros(6))

    def model_inference(self, data):
        with torch.no_grad():
            action_seq = self.model_workspace.model.predict_action(data)
            action = action_seq['action']
            actions = [a.cpu().numpy() for a in action.squeeze(0)]
            rounded_action = torch.round(action * 100) / 100
            # rospy.loginfo(f"Predicted action sequence: {rounded_action}")

        # clear buffer so we never execute an old action sequence
        self.action_buffer = []
        self.action_buffer.extend(actions)
    
    def send_safe_action(self, action):
        # TODO: also set certain vels to 0 if close to boundaries.
        action[:3] = np.clip(action[:3], -0.1, 0.1)  # xyz limits
        action[3:6] = np.deg2rad(np.clip(action[3:6], -5, 5))
        action[6] = np.clip(action[6], 0, 1) * 0.5 # gripper position, scale down to not crush stuff 
        rospy.loginfo(f"Sending action: {action}")
        self.robot.send_twist_topic(action[:6])
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
            next_action = self.action_buffer.pop(0)
            self.send_safe_action(next_action)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("flow_inference_node")
    flow_inference = FlowInference()
    rospy.spin()