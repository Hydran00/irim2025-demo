import argparse
import numpy as np
import open3d as o3d
from convert import SMPL_TO_MOCAP
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm
import random

GLOBAL_TRANS=np.eye(4)

def procrustes_rigid(A, B):
    """
    Rigid Procrustes alignment (rotation + translation) of B to A using scipy's orthogonal_procrustes.
    No scaling applied.
    Returns transformed B.
    """
    # Center both sets
    mu_A = A.mean(axis=0)
    mu_B = B.mean(axis=0)
    AA = A - mu_A
    BB = B - mu_B

    # Compute optimal rotation
    R, _ = orthogonal_procrustes(BB, AA)

    # Apply rotation and translation
    B_aligned = BB @ R + mu_A
    return B_aligned

def compute_PJPE(j1, j2):
    """
    Compute Per Joint Position Error between two joints
    """
    return np.linalg.norm(j1 - j2)

def check_valid(name, arr):
    if not np.all(np.isfinite(arr)):
        print(f"[WARN] {name} contains NaN or Inf!")
        bad_idx = np.where(~np.isfinite(arr))
        print(f" -> Bad indices: {bad_idx}")
        return False
    return True


def combine_mocap_and_smpl_skeleton(mocap_points, smpl_points, zed_points):
    combined_pc = o3d.geometry.PointCloud()
    points, colors = [], []
    mocap_point = []
    smpl_point = []
    zed_point = []
    for i in range(len(mocap_points)):
        mocap_point = mocap_points[i]
        smpl_point = smpl_points[i]
        zed_point = zed_points[i]

        points.extend([mocap_point, smpl_point, zed_point])
        colors.extend([[0, 1, 0], [1, 0.5, 0], [0, 0, 1]])

    combined_pc.points = o3d.utility.Vector3dVector(np.array(points))
    combined_pc.colors = o3d.utility.Vector3dVector(np.array(colors))
    return combined_pc

def show_colored_joints(mocap_points, smpl_points, zed_points):
    combined_pc = o3d.geometry.PointCloud()
    points, colors = [], []
    mocap_point = []
    smpl_point = []
    zed_point = []
    print("Shapes:", len(mocap_points), len(smpl_points), len(zed_points))
    for keypoint in range(len(smpl_points)):
        color = [random.random(), random.random(), random.random()]
        # MOCAP
        mocap_point = mocap_points[keypoint]
        colors.append(color)
        points.append(mocap_point)
        # SMPL 
        smpl_point = smpl_points[keypoint]
        colors.append(color)
        points.append(smpl_point)
        # ZED
        # zed_point = zed_points[keypoint]
        # colors.append(color)
        # points.append(zed_point)
    combined_pc.points = o3d.utility.Vector3dVector(np.array(points))
    combined_pc.colors = o3d.utility.Vector3dVector(np.array(colors))
    return combined_pc

def visualize_npz(data, frame_idx: int = 0,  visualize: bool = False):

    # Convert to numeric arrays
    mocap_points = np.array(data["mocap_skeleton"][frame_idx], dtype=np.float64)
    smpl_points_arr = np.array(data["smpl_keypoints"][frame_idx], dtype=np.float64)
    zed_points_arr = np.array(data["zed_keypoints"][frame_idx], dtype=np.float64)
    cloud_points = np.array(data["cloud"][frame_idx], dtype=np.float64)

    # visualize raw points
    if visualize:
        mocap_pc = o3d.geometry.PointCloud()
        mocap_pc.points = o3d.utility.Vector3dVector(mocap_points)
        mocap_pc.paint_uniform_color([0, 1, 0])  # Green  
        smpl_pc = o3d.geometry.PointCloud()
        smpl_pc.points = o3d.utility.Vector3dVector(smpl_points_arr)
        smpl_pc.paint_uniform_color([1, 0.5, 0])  # Orange
        zed_pc = o3d.geometry.PointCloud()
        zed_pc.points = o3d.utility.Vector3dVector(zed_points_arr)
        zed_pc.paint_uniform_color([0, 0, 1])  # Blue
        o3d.visualization.draw_geometries([mocap_pc, smpl_pc, zed_pc], window_name="Raw Mocap (Green), SMPL (Orange), ZED (Blue)")
    # Prepare subsets for alignment
    mocap_indices = list(SMPL_TO_MOCAP.values())
    smpl_indices = list(SMPL_TO_MOCAP.keys())

    mocap_subset = mocap_points[mocap_indices]
    smpl_subset = smpl_points_arr[smpl_indices]
    zed_subset = zed_points_arr[smpl_indices]

    print(f"Shapes: mocap {mocap_subset.shape}, smpl {smpl_subset.shape}, zed {zed_subset.shape}")


    # Align SMPL and ZED to Mocap
    if not (check_valid("mocap_subset", mocap_subset) 
        and check_valid("smpl_subset", smpl_subset) 
        and check_valid("zed_subset", zed_subset)):
        print(f"Skipping frame {frame_idx} due to invalid data")
        return (None, None, None)
    smpl_aligned_subset = procrustes_rigid(mocap_subset, smpl_subset)
    zed_aligned_subset = procrustes_rigid(mocap_subset, zed_subset)

    # Insert aligned points back into full arrays
    smpl_aligned = smpl_aligned_subset#smpl_points_arr.copy()
    zed_aligned = zed_aligned_subset#zed_points_arr.copy()
    # for i, idx in enumerate(smpl_indices):
    #     smpl_aligned[idx] = smpl_aligned_subset[i]
    #     zed_aligned[idx] = zed_aligned_subset[i]

    # Open3D visualization
    if visualize:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name="Mocap + SMPL + ZED Point Picking")
        combined_pc = show_colored_joints(mocap_subset, smpl_aligned, zed_aligned)
        vis.add_geometry(combined_pc)
        # Render options
        opt = vis.get_render_option()
        opt.point_size = 10.0
        opt.background_color = np.asarray([0.1, 0.1, 0.1])
        vis.run()
        vis.destroy_window()

    # Point cloud
    cloud_geom = o3d.geometry.PointCloud()
    cloud_geom.points = o3d.utility.Vector3dVector(cloud_points)
    cloud_geom.paint_uniform_color([0.7, 0.7, 0.7])
    # vis.add_geometry(cloud_geom)

    # Combined skeleton
    # merged_geom = combine_mocap_and_smpl_skeleton(mocap_subset, smpl_aligned, zed_aligned)



        # picked_ids = vis.get_picked_points()
        # print("Picked indices:", picked_ids)
        # for idx in picked_ids:
        #     if idx < len(mocap_points):
        #         print(f"Picked Mocap point {idx}: {mocap_points[idx]}")
        #     elif idx < len(mocap_points) + len(smpl_aligned):
        #         smpl_idx = idx - len(mocap_points)
        #         print(f"Picked SMPL point {smpl_idx}: {smpl_aligned[smpl_idx]}")
        #     else:
        #         zed_idx = idx - len(mocap_points) - len(smpl_aligned)
        #         print(f"Picked ZED point {zed_idx}: {zed_aligned[zed_idx]}")

    return (smpl_aligned, zed_aligned, mocap_subset)

def main():
    parser = argparse.ArgumentParser(description="Visualize NPZ mocap + SMPL + ZED with Procrustes alignment")
    parser.add_argument("--npz_path", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    args = parser.parse_args()
    data = np.load(args.npz_path, allow_pickle=True)
    NUM_FRAMES = 90
    NUM_JOINTS = len(SMPL_TO_MOCAP)
    zed_vs_gt = [[0 for j in range(NUM_JOINTS)] for i in range(NUM_FRAMES)]
    smpl_vs_gt = [[0 for j in range(NUM_JOINTS)] for i in range(NUM_FRAMES)]
    keys = list(SMPL_TO_MOCAP.keys())
    valid_frames = 0
    for i in range(NUM_FRAMES):
        print(f"Processing frame {i}")
        (smpl_points, zed_points, mocap_points) = visualize_npz(data, frame_idx=i+50, visualize=False)
        if smpl_points is None or zed_points is None or mocap_points is None:
            continue
        if smpl_points[0].shape != mocap_points[0].shape:
            continue
        if zed_points[0].shape != mocap_points[0].shape:
            continue
        for j in range(NUM_JOINTS):
            zed_vs_gt[i][j] = compute_PJPE(zed_points[j], mocap_points[j])
            smpl_vs_gt[i][j] = compute_PJPE(smpl_points[j], mocap_points[j])
        print(f"Zed vs SMPL against gt in frame {i}: ")

        for j in range(NUM_JOINTS):
            print(f"Joint {keys[j]}: ", end="")
            print(f"{zed_vs_gt[i][j]:.2f} ", end=" ")
            print(f"{smpl_vs_gt[i][j]:.2f} ")
        valid_frames += 1

    # print("MPJPE per joint:")
    # for j in range(NUM_JOINTS):
    #     zed_joint_errors = [zed_vs_gt[i][j] for i in range(NUM_FRAMES)]
    #     smpl_joint_errors = [smpl_vs_gt[i][j] for i in range(NUM_FRAMES)]
    #     print(f"Joint {j}: ZED vs GT: {np.mean(zed_joint_errors):.2f}, SMPL vs GT: {np.mean(smpl_joint_errors):.2f}")

    print("Overall MPJPE:")
    all_zed_errors = [zed_vs_gt[i][j] for i in range(NUM_FRAMES) for j in range(NUM_JOINTS)]
    all_smpl_errors = [smpl_vs_gt[i][j] for i in range(NUM_FRAMES) for j in range(NUM_JOINTS)]
    print(f"ZED vs GT: {np.mean(all_zed_errors):.2f}, SMPL vs GT: {np.mean(all_smpl_errors):.2f}")
if __name__ == "__main__":
    main()
