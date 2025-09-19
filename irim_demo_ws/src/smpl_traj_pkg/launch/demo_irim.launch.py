from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # zed node
    # --ros-args -p publish_point_cloud:=true --ros-args -p publish_image:=false
    zed_node = Node(
        package='smpl_ros',
        executable='zed_smpl_tracking',
        output='screen',
        parameters=[
            {'publish_point_cloud': True},
            {'publish_image': False},  # PERFORMANCE
        ],
    )


    # ur driver
    ur_launch = IncludeLaunchDescription(
        launch_description_source=get_package_share_directory('easy_ur_control') + '/launch/easy_ur_launcher.launch.py',
        launch_arguments={
            'robot_ip': '192.168.100.10',
            'ur_type': 'ur3e',
            'ctrl': 'cartesian_compliance_controller'
        }.items()
    )
    # robot-camera transformation
    # ros2 launch easy_handeye2 publish.launch.py 
    camera_robot_tf_pub = IncludeLaunchDescription(
        launch_description_source=get_package_share_directory('easy_handeye2') + '/launch/publish.launch.py',
    )
    #ros2 launch smpl_ros smpl_ros_visualizer.launch.py model_path:=torchure_smplx/SMPL_MALE.npz frame_id:=cam1_36560304 
    smpl_vis = IncludeLaunchDescription(
        launch_description_source=get_package_share_directory('smpl_ros') + '/launch/smpl_ros_visualizer.launch.py',
        launch_arguments={
            'model_path': os.path.expanduser('~/torchure_smplx/SMPL_MALE.npz'),
            'frame_id': 'cam1_36560304'
        }.items()
    )  
    ld = LaunchDescription()
    ld.add_action(ur_launch)
    ld.add_action(camera_robot_tf_pub)
    ld.add_action(zed_node)
    ld.add_action(smpl_vis)
    return ld