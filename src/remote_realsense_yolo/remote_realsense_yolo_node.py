#!/usr/bin/env python3

"""
Remote RealSense YOLO Node

This node handles remote RealSense camera data and performs YOLO object detection.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy
import cv2
import numpy as np


class RemoteRealSenseYOLONode(Node):
    """
    ROS2 node for remote RealSense camera with YOLO object detection.
    """

    def __init__(self):
        super().__init__('remote_realsense_yolo_node')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/camera/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('yolo_model_path', '/path/to/yolo/model.weights')
        self.declare_parameter('yolo_config_path', '/path/to/yolo/model.cfg')
        self.declare_parameter('yolo_classes_path', '/path/to/yolo/classes.names')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('nms_threshold', 0.4)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('visualization_topic', '/detections/visualization')
        self.declare_parameter('max_detection_fps', 30.0)
        self.declare_parameter('enable_gpu', False)
        self.declare_parameter('show_image_window', True)
        self.declare_parameter('window_name', 'Remote RealSense YOLO')
        self.declare_parameter('window_fullscreen', False)
        self.declare_parameter('window_resizable', True)
        self.declare_parameter('qos_reliability', 'reliable')
        self.declare_parameter('qos_durability', 'transient_local')
        self.declare_parameter('qos_depth', 10)
        
        # Get parameter values
        self.image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        self.use_compressed = self.get_parameter('use_compressed').get_parameter_value().bool_value
        self.show_image_window = self.get_parameter('show_image_window').get_parameter_value().bool_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.window_fullscreen = self.get_parameter('window_fullscreen').get_parameter_value().bool_value
        self.window_resizable = self.get_parameter('window_resizable').get_parameter_value().bool_value
        self.qos_reliability = self.get_parameter('qos_reliability').get_parameter_value().string_value
        self.qos_durability = self.get_parameter('qos_durability').get_parameter_value().string_value
        self.qos_depth = self.get_parameter('qos_depth').get_parameter_value().integer_value
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create publishers
        self.detection_pub = self.create_publisher(
            Detection2DArray,
            'detections',
            10
        )
        
        # Create QoS profile for compressed image topic
        if self.use_compressed:
            # Map string parameters to QoS policies
            reliability_policy = QoSReliabilityPolicy.RELIABLE if self.qos_reliability == 'reliable' else QoSReliabilityPolicy.BEST_EFFORT
            durability_policy = QoSDurabilityPolicy.TRANSIENT_LOCAL if self.qos_durability == 'transient_local' else QoSDurabilityPolicy.VOLATILE
            
            # QoS profile for compressed image
            qos_profile = QoSProfile(
                reliability=reliability_policy,
                durability=durability_policy,
                depth=self.qos_depth
            )
            
            self.get_logger().info(f'Using QoS: reliability={self.qos_reliability}, durability={self.qos_durability}, depth={self.qos_depth}')
            
            self.image_sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_callback,
                qos_profile
            )
        else:
            # Default QoS for regular image
            self.image_sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                10
            )
        
        # Initialize YOLO (placeholder - you'll need to implement actual YOLO)
        self.yolo_initialized = False
        self.init_yolo()
        
        # Initialize image window if enabled
        if self.show_image_window:
            # Set window flags based on configuration
            window_flags = cv2.WINDOW_AUTOSIZE
            if self.window_resizable:
                window_flags = cv2.WINDOW_NORMAL
            
            cv2.namedWindow(self.window_name, window_flags)
            
            # Set fullscreen if requested
            if self.window_fullscreen:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            self.get_logger().info(f'Image display window created: {self.window_name} (fullscreen: {self.window_fullscreen}, resizable: {self.window_resizable})')
        
        self.get_logger().info('Remote RealSense YOLO node started')

    def init_yolo(self):
        """
        Initialize YOLO model.
        This is a placeholder - implement your YOLO initialization here.
        """
        try:
            # TODO: Initialize your YOLO model here
            # Example:
            # self.net = cv2.dnn.readNet('yolo.weights', 'yolo.cfg')
            # self.classes = self.load_classes('coco.names')
            self.yolo_initialized = True
            self.get_logger().info('YOLO model initialized successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize YOLO: {str(e)}')
            self.yolo_initialized = False

    def image_callback(self, msg):
        """
        Callback function for incoming camera images.
        
        Args:
            msg: sensor_msgs/Image or sensor_msgs/CompressedImage message
        """
        try:
            # Convert ROS image to OpenCV format based on message type
            if self.use_compressed:
                # Handle compressed image
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    self.get_logger().warn('Failed to decode compressed image')
                    return
            else:
                # Handle regular image
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Display image in window if enabled
            if self.show_image_window:
                self.display_image(cv_image)
            
            if self.yolo_initialized:
                # Perform YOLO detection
                detections = self.perform_yolo_detection(cv_image)
                
                # Publish detections
                self.publish_detections(detections, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def perform_yolo_detection(self, image):
        """
        Perform YOLO object detection on the input image.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            list: List of detection results
        """
        # TODO: Implement actual YOLO detection
        # This is a placeholder that returns empty detections
        detections = []
        
        # Example detection (remove this when implementing real YOLO):
        # if len(image.shape) == 3:
        #     height, width = image.shape[:2]
        #     # Add your YOLO inference code here
        #     pass
        
        return detections

    def publish_detections(self, detections, header):
        """
        Publish detection results as ROS2 message.
        
        Args:
            detections: List of detection results
            header: ROS2 message header
        """
        detection_array = Detection2DArray()
        detection_array.header = header
        
        for detection in detections:
            detection_2d = Detection2D()
            detection_2d.header = header
            
            # TODO: Fill in detection data based on your YOLO output format
            # Example:
            # detection_2d.bbox.center.position.x = detection['center_x']
            # detection_2d.bbox.center.position.y = detection['center_y']
            # detection_2d.bbox.size_x = detection['width']
            # detection_2d.bbox.size_y = detection['height']
            
            # hypothesis = ObjectHypothesisWithPose()
            # hypothesis.hypothesis.class_id = detection['class_id']
            # hypothesis.hypothesis.score = detection['confidence']
            # detection_2d.results.append(hypothesis)
            
            detection_array.detections.append(detection_2d)
        
        self.detection_pub.publish(detection_array)
    
    def display_image(self, cv_image):
        """
        Display the image in a window.
        
        Args:
            cv_image: OpenCV image (numpy array)
        """
        try:
            # Add text overlay with image info
            display_image = cv_image.copy()
            height, width = display_image.shape[:2]
            
            # Add image info text
            info_text = f"Size: {width}x{height} | Topic: {self.image_topic}"
            cv2.putText(display_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add transport type info
            transport_text = f"Transport: {'compressed' if self.use_compressed else 'raw'}"
            cv2.putText(display_image, transport_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add window mode info
            mode_text = f"Mode: {'Fullscreen' if self.window_fullscreen else 'Window'}"
            cv2.putText(display_image, mode_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add keyboard controls info
            controls_text = "Controls: F=Fullscreen, R=Resizable, Q/ESC=Quit"
            cv2.putText(display_image, controls_text, (10, height - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the image
            cv2.imshow(self.window_name, display_image)
            
            # Handle window events (non-blocking)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC key
                self.get_logger().info('Window closed by user')
                cv2.destroyAllWindows()
                self.show_image_window = False
            elif key == ord('f'):  # 'f' key to toggle fullscreen
                self.toggle_fullscreen()
            elif key == ord('r'):  # 'r' key to toggle resizable
                self.toggle_resizable()
                
        except Exception as e:
            self.get_logger().error(f'Error displaying image: {str(e)}')
    
    def toggle_fullscreen(self):
        """
        Toggle fullscreen mode for the display window.
        """
        try:
            if self.show_image_window:
                self.window_fullscreen = not self.window_fullscreen
                if self.window_fullscreen:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    self.get_logger().info('Window set to fullscreen mode')
                else:
                    cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    self.get_logger().info('Window set to normal mode')
        except Exception as e:
            self.get_logger().error(f'Error toggling fullscreen: {str(e)}')
    
    def toggle_resizable(self):
        """
        Toggle resizable property for the display window.
        """
        try:
            if self.show_image_window:
                self.window_resizable = not self.window_resizable
                # Note: Window flags cannot be changed after creation in OpenCV
                # This is more of a status indicator for future reference
                self.get_logger().info(f'Window resizable property set to: {self.window_resizable}')
        except Exception as e:
            self.get_logger().error(f'Error toggling resizable: {str(e)}')


def main(args=None):
    """
    Main function to start the node.
    """
    rclpy.init(args=args)
    
    node = RemoteRealSenseYOLONode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
