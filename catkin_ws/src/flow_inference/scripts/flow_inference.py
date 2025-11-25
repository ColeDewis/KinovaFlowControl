import rospy
from kortex_bringup import KinovaGen3
from sensor_msgs.msg import Image, JointState, PointCloud2


class FlowInference:
    def __init__(self):
        self.robot = KinovaGen3()

        self.pc_subscriber = rospy.Subscriber(
            "/camera1/depth/color/points", PointCloud2, self.point_cloud_callback
        )

    def point_cloud_callback(self, msg):
        # Process the point cloud data
        pass
