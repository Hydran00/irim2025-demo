#!/usr/bin/env python3
import numpy as np
# import open3d as o3d
import trimesh
import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion, TransformStamped, PoseStamped
import tf2_geometry_msgs
from visualization_msgs.msg import MarkerArray, Marker

from scipy.spatial.transform import Rotation as R
from roboticstoolbox.tools.trajectory import ctraj
from spatialmath import SE3

import tf2_ros

# ---------------------------
# Global list of "offline vertices" to visit
# OFFLINE_VERTICES_IDX = [5324, 5371, 5677, 5635, 5608, 6016, 5635, 6102, 6584, 6451, 6584, 6746, 6677, 6792, 6802, 6855]
OFFLINE_VERTICES_IDX = [5319, 5324, 5590, 6362, 6768]
# ---------------------------


class SMPLTrajNode(Node):
    def __init__(self):
        super().__init__("smpl_traj_node")
        # self.declare_parameter("steps_per_segment", 300)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_rate", 500.0)  # Hz for PoseStamped stream

        # self.steps_per_segment = self.get_parameter("steps_per_segment").value
        self.steps_per_segment = 200  # will be updated dynamically
        self.frame_id = self.get_parameter("frame_id").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.publish_rate = self.get_parameter("publish_rate").value

        # Publish PoseStamped instead of PoseArray
        self.pose_pub = self.create_publisher(PoseStamped, "/target_frame", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/target_pose_markers", 10)
        self.marker_sub = self.create_subscription(
            MarkerArray, "/smpl_markers", self.marker_callback, 10
        )
        self.srv = self.create_service(Trigger, "/compute_traj", self.handle_trigger)

        # Timer for streaming poses
        self.traj_buffer = []
        self.traj_index = 0
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.last_vertices = None
        self.last_triangles = None
        self.is_moving = False  # True when trajectory is being executed
        self.get_logger().info("smpl_traj_node started.")

    def marker_callback(self, msg: MarkerArray):
        if self.is_moving:
            return
        for marker in msg.markers:
            if marker.ns != "smpl_mesh":
                continue
            pts = np.array([[p.x +0.5, p.y+0.5, p.z+0.5] for p in marker.points])
            if pts.size == 0:
                continue

            unique_pts, inverse = np.unique(
                np.round(pts, 8), axis=0, return_inverse=True
            )
            triangles = []
            for i in range(0, len(inverse) - (len(inverse) % 3), 3):
                triangles.append([inverse[i], inverse[i + 1], inverse[i + 2]])
            self.last_vertices = unique_pts
            self.last_triangles = np.array(triangles, dtype=int)
            self.get_logger().info(
                f"Received smpl_mesh with {self.last_vertices.shape[0]} vertices"
            )

    def build_mesh(self, vertices, triangles=None):
        if triangles is None:
            mesh = trimesh.Trimesh(vertices=vertices, process=False)
        else:
            mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
            # write mesh to file for debugging
            # mesh.export("smpl_mesh.ply")
            # self.get_logger().info("Exported smpl_mesh.ply for debugging.")
        return mesh

    def compute_inward_normals(self, mesh):
        normals = mesh.vertex_normals
        # create copy to avoid modifying original normals
        normals = np.array(normals)
        verts = mesh.vertices
        centroid = verts.mean(axis=0)
        for i in range(len(verts)):
            to_cent = centroid - verts[i]
            if np.dot(normals[i], to_cent) < 0:
                normals[i] = -normals[i]
        return normals

    def pose_from_vertex_normal(self, pos, normal, centroid=None):
        """
        Build orientation from inward normal and 'down' reference.
        Z axis -> inward normal
        X axis -> projection of world-down (0,0,-1) onto tangent plane
        Y axis -> right-hand rule
        """
        # Ensure inward normal
        z_axis = normal / np.linalg.norm(normal)
        if centroid is not None:
            to_cent = (centroid - pos) / np.linalg.norm(centroid - pos)
            if np.dot(z_axis, to_cent) < 0:  # flip if pointing outward
                z_axis = -z_axis

        # Reference down vector
        down = np.array([0.0, 0.0, -1.0])

        # Project "down" onto tangent plane (orthogonal to z)
        x_axis = down - np.dot(down, z_axis) * z_axis
        if np.linalg.norm(x_axis) < 1e-6:
            # If normal is parallel to down, pick arbitrary orthogonal axis
            x_axis = np.array([1.0, 0.0, 0.0])
        x_axis /= np.linalg.norm(x_axis)

        # Right-hand rule for y
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        Rm = np.column_stack((x_axis, y_axis, z_axis))
        return Rm, pos

    def get_current_ee_pose(self):
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.frame_id, self.ee_frame, rclpy.time.Time()
            )
            t = tf.transform.translation
            q = tf.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = [t.x, t.y, t.z]
            return SE3(T)
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

    def compute_traj(self, vertices, triangles=None):
        # Build mesh and compute inward normals
        mesh = self.build_mesh(vertices, triangles)
        normals = self.compute_inward_normals(mesh)

        # Select offline vertices
        sel_idx = OFFLINE_VERTICES_IDX
        sel_idx = [i for i in sel_idx if i < len(vertices)]

        # Compute poses for each selected vertex
        centroid = vertices.mean(axis=0)
        poses = [self.pose_from_vertex_normal(vertices[i], normals[i], centroid) for i in sel_idx]
        print("Looking at poses:")
        for i, (Rm, pos) in enumerate(poses):
            print(f"Pose {i}: Pos: {pos}, Normal (Z): {Rm[:,2]}")

        # Get current EE pose as start
        ee_pose = self.get_current_ee_pose()
        if ee_pose is None:
            self.get_logger().warn("EE pose not available, starting from first waypoint.")
        
        # Build waypoints SE3 list
        waypoints = []
        if ee_pose is not None:
            waypoints.append(ee_pose)  # prepend current EE pose
        for Rm, pos in poses:
            T = np.eye(4)
            T[:3, :3] = Rm
            T[:3, 3] = pos
            waypoints.append(SE3(T))

        traj = []
        # Generate point-to-point trajectories
        for i in range(len(waypoints) - 1):
            se3_start = waypoints[i]
            se3_end = waypoints[i + 1]

            # Compute Cartesian distance
            cart_distance = np.linalg.norm(se3_start.t - se3_end.t)
            # Compute steps based on distance (at least 50 steps)
            steps = max(int(self.publish_rate * cart_distance / 0.05), 400)

            self.get_logger().info(f"Segment {i}->{i+1}: distance {cart_distance:.3f} m, steps {steps}")
            Tg = ctraj(se3_start, se3_end, t=steps)
            for T in Tg:
                if np.linalg.norm(T.t) > 0.001:  # skip zero points
                    traj.append(T)
        self.get_logger().info(f"Computed trajectory with {len(traj)} poses.")
        return traj
    def publish_markers(self, traj):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        traj = [(T.R, T.t) for T in traj]

        # LINE_STRIP marker to show the path
        line = Marker()
        line.header.frame_id = self.frame_id
        line.header.stamp = stamp
        line.action = Marker.ADD
        line.ns = "traj_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.01  # line width
        line.color.a = 1.0
        line.color.r, line.color.g, line.color.b = 1.0, 0.0, 0.0
        line.points = [Point(x=float(pos[0]), y=float(pos[1]), z=float(pos[2])) for _, pos in traj]
        ma.markers.append(line)

        self.marker_pub.publish(ma)
        self.get_logger().info(
            f"Published LINE_STRIP for {len(traj)} poses to /target_pose_markers"
        )

    # --- Timer publishes next PoseStamped from buffer ---
    def timer_callback(self):
        if not self.traj_buffer:
            return
        if self.traj_index >= len(self.traj_buffer):
            self.traj_index = 0
            self.traj_buffer = []
            self.is_moving = False
            self.get_logger().info("Trajectory completed.")
            return  # finished trajectory, stop publishing
        self.is_moving = True
        T = self.traj_buffer[self.traj_index]
        quat = R.from_matrix(T.R).as_quat()

        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        # transform back to base_link
        ps.header.frame_id = self.frame_id
        ps.pose.position = Point(x=T.t[0], y=T.t[1], z=T.t[2])
        ps.pose.orientation = Quaternion(
            x=quat[0], y=quat[1], z=quat[2], w=quat[3]
        )
        pose_in_robot_base = self.tf_buffer.transform(ps, self.base_frame, timeout=rclpy.duration.Duration(seconds=1.0))

        self.pose_pub.publish(pose_in_robot_base)
        self.traj_index += 1

    # --- Service handler ---
    def handle_trigger(self, request, response):
        self.get_logger().info("Received /compute_traj request.")
        verts, tris = self.last_vertices, self.last_triangles
        if verts is None:
            response.success = False
            response.message = "No vertices available in /smpl_markers."
            return response

        traj = self.compute_traj(verts, tris)
        if not traj:
            response.success = False
            response.message = "Trajectory generation failed."
            return response

        # Reset trajectory buffer
        self.traj_buffer = traj
        self.traj_index = 0

        # Publish visualization markers
        self.publish_markers(traj)

        response.success = True
        response.message = f"Streaming trajectory with {len(traj)} poses."
        return response


def main(args=None):
    rclpy.init(args=args)
    print("Starting smpl_traj_node...")
    node = SMPLTrajNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
