from setuptools import setup

package_name = 'remote_realsense_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/remote_realsense_yolo.launch.py']),
        ('share/' + package_name + '/config', ['config/remote_realsense_yolo.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='ROS2 package for remote RealSense camera with YOLO object detection',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'remote_realsense_yolo_node = remote_realsense_yolo.remote_realsense_yolo_node:main',
        ],
    },
)
