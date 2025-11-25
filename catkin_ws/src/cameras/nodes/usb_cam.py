import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraPublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node("camera_publisher", anonymous=True)

        # Create a publisher for the camera feed
        self.image_pub = rospy.Publisher(
            "/camera1/color/image_raw", Image, queue_size=10
        )

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Open the camera (0 for default camera)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera.")
            rospy.signal_shutdown("Camera not available.")

        rospy.loginfo("Started camera.")

    def publish_frames(self):
        rate = rospy.Rate(10)  # Publish at 10 Hz
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to read frame from camera.")
                continue

            # Convert the OpenCV frame to a ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")

            # Publish the image
            self.image_pub.publish(ros_image)

            rate.sleep()

    def cleanup(self):
        # Release the camera resource
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        camera_publisher = CameraPublisher()
        camera_publisher.publish_frames()
    except rospy.ROSInterruptException:
        pass
    finally:
        camera_publisher.cleanup()
