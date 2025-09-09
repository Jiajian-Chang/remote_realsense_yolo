#!/usr/bin/env python3

"""
Launch file for Remote RealSense YOLO node.
"""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate launch description for the remote RealSense YOLO node.
    """
    
    # Get package directory
    pkg_share = get_package_share_directory('remote_realsense_yolo')
    
    # Declare launch arguments
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=os.path.join(pkg_share, 'config', 'remote_realsense_yolo.yaml'),
        description='Path to the configuration file'
    )
    
    log_level_arg = DeclareLaunchArgument(
        'log_level',
        default_value='info',
        description='Log level (debug, info, warn, error)'
    )
    
    # Create the node
    remote_realsense_yolo_node = Node(
        package='remote_realsense_yolo',
        executable='remote_realsense_yolo_node.py',
        name='remote_realsense_yolo_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )
    
    return LaunchDescription([
        config_file_arg,
        log_level_arg,
        remote_realsense_yolo_node,
    ])
