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
from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path


class RemoteRealSenseYOLONode(Node):
    """
    ROS2 node for remote RealSense camera with YOLO object detection.
    """

    def __init__(self):
        super().__init__('remote_realsense_yolo_node')
        
        # Declare parameters with default values
        self.declare_parameter('image_topic', '/camera/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('yolo_model', 'yolo11n.pt')  # Ultralytics YOLO model
        self.declare_parameter('yolo_model_cache_dir', '~/.cache/ultralytics')  # Model cache directory
        self.declare_parameter('yolo_confidence', 0.5)
        self.declare_parameter('yolo_iou_threshold', 0.4)
        self.declare_parameter('detect_person_only', True)  # Focus on person detection
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
        
        # Initialize YOLO model
        self.yolo_initialized = False
        self.yolo_model = None
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
        Initialize Ultralytics YOLO model with persistent cache.
        """
        try:
            # Get YOLO model and cache directory parameters
            yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
            cache_dir = self.get_parameter('yolo_model_cache_dir').get_parameter_value().string_value
            
            # Expand user path and create cache directory if it doesn't exist
            cache_path = Path(cache_dir).expanduser()
            cache_path.mkdir(parents=True, exist_ok=True)
            
            # Set environment variable for Ultralytics cache
            os.environ['ULTRALYTICS_CACHE_DIR'] = str(cache_path)
            
            self.get_logger().info(f'Using YOLO model cache directory: {cache_path}')
            self.get_logger().info(f'Loading YOLO model: {yolo_model}')
            
            # Load YOLO model (will download to cache if not present)
            self.yolo_model = YOLO(yolo_model)
            
            self.yolo_initialized = True
            self.get_logger().info(f'YOLO model initialized successfully: {yolo_model}')
            
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
            # Debug: Log message info
            self.get_logger().debug(f'Received image message: type={type(msg).__name__}, '
                                  f'header.stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec}, '
                                  f'header.frame_id={msg.header.frame_id}')
            
            # Convert ROS image to OpenCV format based on message type
            if self.use_compressed:
                # Handle compressed image
                self.get_logger().debug(f'Processing compressed image: data_size={len(msg.data)} bytes, '
                                      f'format={msg.format}')
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if cv_image is None:
                    self.get_logger().warn('Failed to decode compressed image')
                    return
                self.get_logger().debug(f'Decoded compressed image: shape={cv_image.shape}, dtype={cv_image.dtype}')
            else:
                # Handle regular image
                self.get_logger().debug(f'Processing regular image: width={msg.width}, height={msg.height}, '
                                      f'encoding={msg.encoding}, step={msg.step}')
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.get_logger().debug(f'Converted regular image: shape={cv_image.shape}, dtype={cv_image.dtype}')
            
            # Perform YOLO detection if initialized
            detections = []
            if self.yolo_initialized:
                self.get_logger().debug('Starting YOLO detection...')
                detections = self.perform_yolo_detection(cv_image)
                self.get_logger().debug(f'YOLO detection completed: found {len(detections)} objects')
                
                # Publish detections
                self.publish_detections(detections, msg.header)
                self.get_logger().debug(f'Published {len(detections)} detections')
                
                # Log detection results
                if detections:
                    person_count = sum(1 for det in detections if det['class_name'] == 'person')
                    self.get_logger().info(f'Detected {person_count} person(s) and {len(detections)} total objects')
                    # Debug: Log individual detections
                    for i, det in enumerate(detections):
                        self.get_logger().debug(f'Detection {i}: {det["class_name"]} '
                                              f'(conf={det["confidence"]:.3f}, '
                                              f'bbox=({det["bbox"]["x1"]:.1f},{det["bbox"]["y1"]:.1f},'
                                              f'{det["bbox"]["x2"]:.1f},{det["bbox"]["y2"]:.1f}))')
            else:
                self.get_logger().debug('YOLO not initialized, skipping detection')
            
            # Display image in window if enabled (with detections overlay)
            if self.show_image_window:
                self.get_logger().debug('Displaying image with detections overlay')
                self.display_image(cv_image, detections)
            else:
                self.get_logger().debug('Image window disabled, skipping display')
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}', exc_info=True)

    def perform_yolo_detection(self, image):
        """
        Perform YOLO object detection on the input image using Ultralytics YOLO.
        
        Args:
            image: OpenCV image (numpy array)
            
        Returns:
            list: List of detection results
        """
        detections = []
        
        if not self.yolo_initialized or self.yolo_model is None:
            return detections
        
        try:
            # Get detection parameters
            confidence = self.get_parameter('yolo_confidence').get_parameter_value().double_value
            iou_threshold = self.get_parameter('yolo_iou_threshold').get_parameter_value().double_value
            detect_person_only = self.get_parameter('detect_person_only').get_parameter_value().bool_value
            
            # Run YOLO inference
            results = self.yolo_model(image, conf=confidence, iou=iou_threshold, verbose=False)
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Get box coordinates and confidence
                        box = boxes.xyxy[i].cpu().numpy()  # x1, y1, x2, y2
                        conf = boxes.conf[i].cpu().numpy()
                        cls = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_model.names[cls]
                        
                        # Filter for person detection if requested
                        if detect_person_only and class_name != 'person':
                            continue
                        
                        # Create detection result
                        detection = {
                            'class_id': cls,
                            'class_name': class_name,
                            'confidence': float(conf),
                            'bbox': {
                                'x1': float(box[0]),
                                'y1': float(box[1]),
                                'x2': float(box[2]),
                                'y2': float(box[3]),
                                'width': float(box[2] - box[0]),
                                'height': float(box[3] - box[1]),
                                'center_x': float((box[0] + box[2]) / 2),
                                'center_y': float((box[1] + box[3]) / 2)
                            }
                        }
                        
                        detections.append(detection)
            
        except Exception as e:
            self.get_logger().error(f'Error in YOLO detection: {str(e)}')
        
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
            
            # Set bounding box information
            bbox = detection['bbox']
            detection_2d.bbox.center.x = float(bbox['center_x'])
            detection_2d.bbox.center.y = float(bbox['center_y'])
            # Optional: set orientation to 0 since we don't estimate rotation
            detection_2d.bbox.center.theta = 0.0
            detection_2d.bbox.size_x = float(bbox['width'])
            detection_2d.bbox.size_y = float(bbox['height'])
            
            # Set object hypothesis (Foxy vision_msgs: id and score fields)
            hypothesis = ObjectHypothesisWithPose()
            # Use class name as id for readability; fallback to class_id string
            class_id_str = str(detection.get('class_name', detection.get('class_id', '')))
            hypothesis.id = class_id_str
            hypothesis.score = float(detection['confidence'])
            detection_2d.results.append(hypothesis)
            
            detection_array.detections.append(detection_2d)
        
        self.detection_pub.publish(detection_array)
    
    def display_image(self, cv_image, detections=None):
        """
        Display the image in a window with optional detection overlays.
        
        Args:
            cv_image: OpenCV image (numpy array)
            detections: List of detection results (optional)
        """
        try:
            # Add text overlay with image info
            display_image = cv_image.copy()
            height, width = display_image.shape[:2]
            
            # Draw detection bounding boxes if provided
            if detections:
                for detection in detections:
                    bbox = detection['bbox']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # Draw bounding box
                    x1, y1 = int(bbox['x1']), int(bbox['y1'])
                    x2, y2 = int(bbox['x2']), int(bbox['y2'])
                    
                    # Choose color based on class (person = green, others = blue)
                    color = (0, 255, 0) if class_name == 'person' else (255, 0, 0)
                    
                    cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label with confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), color, -1)
                    
                    # Draw label text
                    cv2.putText(display_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # # Add image info text
            # info_text = f"Size: {width}x{height} | Topic: {self.image_topic}"
            # cv2.putText(display_image, info_text, (10, 30), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # # Add transport type info
            # transport_text = f"Transport: {'compressed' if self.use_compressed else 'raw'}"
            # cv2.putText(display_image, transport_text, (10, 60), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # # Add window mode info
            # mode_text = f"Mode: {'Fullscreen' if self.window_fullscreen else 'Window'}"
            # cv2.putText(display_image, mode_text, (10, 90), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add detection count info
            if detections:
                person_count = sum(1 for det in detections if det['class_name'] == 'person')
                detection_text = f"Detections: {person_count} person(s), {len(detections)} total"
                cv2.putText(display_image, detection_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # # Add keyboard controls info
            # controls_text = "Controls: F=Fullscreen, R=Resizable, Q/ESC=Quit"
            # cv2.putText(display_image, controls_text, (10, height - 20), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
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
