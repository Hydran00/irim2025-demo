#!/usr/bin/env python3
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node

from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from visualization_msgs.msg import MarkerArray, Marker

from scipy.spatial.transform import Rotation as R
from spatialmath import SE3
from roboticstoolbox.tools.trajectory import ctraj
# ---------------------------
# Global list of "offline vertices" to visit
OFFLINE_VERTICES_IDX = [0, 1, 2]
# ---------------------------

class SMPLTrajNode(Node):
    def __init__(self):
        super().__init__('smpl_traj_node')
        self.declare_parameter('steps_per_segment', 20)
        self.declare_parameter('frame_id', 'world')
        self.get_logger().info("Parameters:")
        self.get_logger().info(" steps_per_segment: %d", self.get_parameter('steps_per_segment').value)
        print(" frame_id:", self.get_parameter('frame_id').value)
        self.steps_per_segment = self.get_parameter('steps_per_segment').value
        self.frame_id = self.get_parameter('frame_id').value

        self.traj_pub = self.create_publisher(PoseArray, '/target_pose', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/target_pose_markers', 10)
        self.marker_sub = self.create_subscription(MarkerArray, '/smpl_markers', self.marker_callback, 10)

        self.srv = self.create_service(Trigger, '/compute_traj', self.handle_trigger)

        self.last_vertices = None
        self.last_triangles = None
        self.get_logger().info('smpl_traj_node started.')

    def marker_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            if marker.ns != 'smpl_mesh':
                continue
            pts = np.array([[p.x, p.y, p.z] for p in marker.points])
            if pts.size == 0:
                continue

            unique_pts, inverse = np.unique(np.round(pts, 8), axis=0, return_inverse=True)
            triangles = []
            for i in range(0, len(inverse) - (len(inverse) % 3), 3):
                triangles.append([inverse[i], inverse[i+1], inverse[i+2]])
            self.last_vertices = unique_pts
            self.last_triangles = np.array(triangles, dtype=int)
            self.get_logger().info(f"Received smpl_mesh with {self.last_vertices.shape[0]} vertices")

    def build_o3d_mesh(self, vertices, triangles=None):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        if triangles is not None and len(triangles) > 0:
            mesh.triangles = o3d.utility.Vector3iVector(triangles)
        return mesh

    def compute_inward_normals(self, mesh):
        mesh.compute_vertex_normals()
        verts = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        centroid = verts.mean(axis=0)
        for i in range(len(verts)):
            to_cent = centroid - verts[i]
            if np.dot(normals[i], to_cent) < 0:
                normals[i] = -normals[i]
        return normals

    def pose_from_vertex_normal(self, pos, normal):
        z_axis = normal / np.linalg.norm(normal)
        tmp = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
        x_axis = np.cross(tmp, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        Rm = np.column_stack((x_axis, y_axis, z_axis))
        return SE3.Rt(Rm, pos)

    def compute_traj(self, vertices, triangles=None):
        mesh = self.build_o3d_mesh(vertices, triangles)
        normals = self.compute_inward_normals(mesh)

        # Use predefined offline vertex indices (global)
        sel_idx = [i for i in OFFLINE_VERTICES_IDX if i < len(vertices)]
        if not sel_idx:
            self.get_logger().error('No valid offline vertices indices found in current mesh.')
            return []

        poses = [self.pose_from_vertex_normal(vertices[i], normals[i]) for i in sel_idx]

        traj = []
        for i in range(len(poses)-1):
            Tg = ctraj(poses[i], poses[i+1], self.steps_per_segment)
            traj.extend(Tg)
        traj.append(poses[-1])
        return traj

    def publish_posearray(self, traj):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = self.frame_id
        for T in traj:
            pos = T.t
            quat = R.from_matrix(T.R).as_quat()
            pose = Pose()
            pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
            pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            pa.poses.append(pose)
        self.traj_pub.publish(pa)
        self.get_logger().info(f'Published {len(pa.poses)} poses to /target_pose')

    def publish_markers(self, traj):
        ma = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        # Arrow markers
        for i, T in enumerate(traj):
            pos = T.t
            quat = R.from_matrix(T.R).as_quat()
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = stamp
            m.ns = "traj_poses"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose.position = Point(x=pos[0], y=pos[1], z=pos[2])
            m.pose.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
            m.scale.x = 0.05
            m.scale.y = 0.01
            m.scale.z = 0.01
            m.color.a = 1.0
            m.color.r, m.color.g, m.color.b = 0.0, 0.5, 1.0
            ma.markers.append(m)

        # Line strip
        line = Marker()
        line.header.frame_id = self.frame_id
        line.header.stamp = stamp
        line.ns = "traj_path"
        line.id = 9999
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.01
        line.color.a = 1.0
        line.color.r, line.color.g, line.color.b = 1.0, 0.0, 0.0
        line.points = [Point(x=T.t[0], y=T.t[1], z=T.t[2]) for T in traj]
        ma.markers.append(line)

        self.marker_pub.publish(ma)
        self.get_logger().info(f'Published markers for {len(traj)} poses to /target_pose_markers')

    def handle_trigger(self, request, response):
        verts, tris = self.last_vertices, self.last_triangles
        if verts is None:
            response.success = False
            response.message = 'No vertices available in /smpl_markers.'
            return response

        traj = self.compute_traj(verts, tris)
        if not traj:
            response.success = False
            response.message = 'Trajectory generation failed.'
            return response

        self.publish_posearray(traj)
        self.publish_markers(traj)

        response.success = True
        response.message = f'Published trajectory with {len(traj)} poses.'
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
