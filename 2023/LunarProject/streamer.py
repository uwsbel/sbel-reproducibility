import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2

class Image_Subscriber(Node):
  def __init__(self):
    super().__init__('Image_Subscriber')
    self.subscription = self.create_subscription(Image, '/sensor_image', self.image_callback, 10)
 
  def image_callback(self, msg):
    print("Received Image")
    image = np.array(msg.data).reshape(msg.height, msg.width, -1)
    image = cv2.flip(image, 0)
    cv2.imshow("Raw Image", image)
    cv2.waitKey(2) 


def main(args=None):
    rclpy.init(args=args)
    image_sub = Image_Subscriber()

    rclpy.spin(image_sub)
    image_sub.destroy_node()
   
    rclpy.shutdown()
 
if __name__ == '__main__':
  main()
