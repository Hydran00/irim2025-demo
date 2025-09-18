import numpy as np
import open3d as o3d
from .ros_utils import pointcloud2_to_xyz_array  # if you keep ROS helpers separate


def visualize_synced_data(synced_data):
    """
    Visualize mocap skeleton points and SMPL keypoints with Open3D.
    Allows Shift + Left Click picking.
    """
    cloud_geom = o3d.geometry.PointCloud()
    mocap_geom = o3d.geometry.PointCloud()
    smpl_geom = o3d.geometry.PointCloud()

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Point Picking")

    for i, entry in enumerate(synced_data):
        print(f"Frame {i}, Timestamp: {entry.timestamp}")

        # --- Point cloud ---
        cloud_points = pointcloud2_to_xyz_array(entry.cloud)
        if cloud_points.shape[0] == 0:
            print(f"[Frame {i}] Empty cloud")
        cloud_geom.points = o3d.utility.Vector3dVector(cloud_points)
        cloud_geom.paint_uniform_color([0.7, 0.7, 0.7])

        # --- Mocap skeleton ---
        mocap_points = np.array([m.translation for m in entry.mocap_skeleton])
        mocap_geom.points = o3d.utility.Vector3dVector(mocap_points)
        mocap_geom.paint_uniform_color([1, 0, 0])

        # --- SMPL keypoints ---
        smpl_points = entry.smpl.keypoints
        if smpl_points is not None and smpl_points.shape[0] > 0:
            smpl_geom.points = o3d.utility.Vector3dVector(smpl_points)
            smpl_geom.paint_uniform_color([0, 0, 1])

        print("SMPL keypoints:", len(smpl_points) if smpl_points is not None else 0)
        print("Mocap markers:", len(mocap_points))

        # Add geometries
        # vis.add_geometry(cloud_geom)  # optional
        vis.add_geometry(mocap_geom)
        vis.add_geometry(smpl_geom)

        print(">>> Shift + Left Click to pick points, then press Q to quit <<<")
        vis.run()

        picked_ids = vis.get_picked_points()
        print("Picked indices:", picked_ids)

        # Combine all points
        all_points = np.vstack([
            np.asarray(mocap_geom.points),
            np.asarray(smpl_geom.points),
        ])

        for idx in picked_ids:
            if idx < len(all_points):
                print(f"Picked point {idx}: {all_points[idx]}")
            else:
                print(f"Picked index {idx} out of range")

    vis.destroy_window()
