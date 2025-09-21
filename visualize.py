import argparse
import numpy as np
from convert import SMPL_TO_MOCAP, SMPL_JOINT_NAMES
from scipy.linalg import orthogonal_procrustes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D projection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import os

os.environ["PYTHONNOUSERSITE"] = "1"
VISUALIZE = False


def procrustes_rigid_pelvis(A, B, pelvis_idx=0):
    """
    Align B to A using rigid Procrustes with pelvis as pivot.
    A, B: (N,3)
    Returns B_aligned
    """
    # shift so pelvis is at origin
    A0 = A - A[pelvis_idx]
    B0 = B - B[pelvis_idx]

    # compute rotation only
    R, _ = orthogonal_procrustes(B0, A0)

    # rotate B and shift back to pelvis of A
    B_aligned = B0 @ R + A[pelvis_idx]
    return B_aligned


def compute_PJPE(j1, j2):
    """Per Joint Position Error"""
    return np.linalg.norm(j1 - j2)


def check_valid(name, arr):
    if not np.all(np.isfinite(arr)):
        print(f"[WARN] {name} contains NaN or Inf!")
        return False
    return True


def plot_skeleton(mocap_points, smpl_points=None, zed_points=None):
    """Plot Mocap, SMPL, and ZED skeletons with Matplotlib 3D"""

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Scatter points
    ax.scatter(
        mocap_points[:, 0], mocap_points[:, 1], mocap_points[:, 2], c="g", label="Mocap"
    )
    if smpl_points is not None:
        ax.scatter(
            smpl_points[:, 0],
            smpl_points[:, 1],
            smpl_points[:, 2],
            c="orange",
            label="SMPL",
        )
    if zed_points is not None:
        ax.scatter(
            zed_points[:, 0],
            zed_points[:, 1],
            zed_points[:, 2],
            c="b",
            label="ZED",
        )

    # Skeleton lines
    mocap_segments = []
    smpl_segments = []
    zed_segments = []

    # for i, parent in SMPL_PARENTS.items():
    #     if parent == -1:  # root, no parent
    #         continue
    #     if i == 21:
    #         continue  # right_hand ignored
    #     mocap_segments.append([mocap_points[i], mocap_points[parent]])
    #     smpl_segments.append([smpl_points[i], smpl_points[parent]])
    #     zed_segments.append([zed_points[i], zed_points[parent]])

    # # Create line collections
    # ax.add_collection3d(Line3DCollection(mocap_segments, colors="g", linewidths=2))
    # ax.add_collection3d(Line3DCollection(smpl_segments, colors="orange", linewidths=2))
    # ax.add_collection3d(Line3DCollection(zed_segments, colors="b", linewidths=2))

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mocap (Green), SMPL (Orange), ZED (Blue)")
    ax.legend()

    fig.set_size_inches(18, 16)
    # set limits based on pelvis (index 0)
    pelvis = mocap_points[0]
    ax.set_xlim(pelvis[0] - 1, pelvis[0] + 1)
    ax.set_ylim(pelvis[1] - 1, pelvis[1] + 1)
    ax.set_zlim(pelvis[2] - 1, pelvis[2] + 1)
    fig.tight_layout()
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def visualize_npz(data, frame_idx: int = 0, visualize: bool = False):
    mocap_points = np.array(data["mocap_skeleton"][frame_idx], dtype=np.float64)
    smpl_points_arr = np.array(data["smpl_keypoints"][frame_idx], dtype=np.float64)
    zed_points_arr = np.array(data["zed_keypoints"][frame_idx], dtype=np.float64)

    mocap_indices = list(SMPL_TO_MOCAP.values())
    smpl_indices = list(SMPL_TO_MOCAP.keys())
    # print("Mocap shape:", mocap_points.shape)
    # print("SMPL shape:", smpl_points_arr.shape)
    # print("ZED shape:", zed_points_arr.shape)
    # idx = [0, 8, 19, 20]
    mocap_subset = mocap_points[mocap_indices][:]
    smpl_subset = smpl_points_arr[smpl_indices][:]
    zed_subset = zed_points_arr[smpl_indices][:]

    if not (
        check_valid("mocap_subset", mocap_subset)
        and check_valid("smpl_subset", smpl_subset)
        and check_valid("zed_subset", zed_subset)
    ):
        print(f"Skipping frame {frame_idx} due to invalid data")
        return (None, None, None)

    if visualize:
        print("Before alignment:")
        plot_skeleton(mocap_subset, smpl_points=smpl_subset, zed_points=zed_subset)

    mocap_subset_cpy = mocap_subset.copy()
    smpl_aligned = procrustes_rigid_pelvis(mocap_subset, smpl_subset)
    zed_aligned = procrustes_rigid_pelvis(mocap_subset, zed_subset)

    # print("Zed point array shape:", zed_subset.shape)
    # print("SMPL point array shape:", smpl_subset.shape)
    # print("Mocap subset shape:", mocap_subset.shape)

    if visualize:
        plot_skeleton(mocap_subset_cpy, smpl_points=smpl_aligned)
        plot_skeleton(mocap_subset_cpy, zed_points=zed_aligned)
    return (smpl_aligned, zed_aligned, mocap_subset)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NPZ mocap + SMPL + ZED with Procrustes alignment (Matplotlib)"
    )
    parser.add_argument("--npz_path", type=str, required=True, help="Path to .npz file")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize")
    args = parser.parse_args()

    data = np.load(args.npz_path, allow_pickle=True)
    NUM_FRAMES = len(data["mocap_skeleton"])
    NUM_JOINTS = len(SMPL_TO_MOCAP)
    zed_vs_gt = [[np.nan for j in range(NUM_JOINTS)] for i in range(NUM_FRAMES)]
    smpl_vs_gt = [[np.nan for j in range(NUM_JOINTS)] for i in range(NUM_FRAMES)]
    keys = list(SMPL_TO_MOCAP.keys())

    for i in range(NUM_FRAMES):
        print(f"Processing frame {i}")
        smpl_points, zed_points, mocap_points = visualize_npz(
            data, frame_idx=i + args.frame, visualize=VISUALIZE
        )
        if smpl_points is None or zed_points is None or mocap_points is None:
            print(f"Skipping frame {i} due to invalid data")
            continue
        # print("MPJPE per joint:")
        for j in range(NUM_JOINTS):
            zed_vs_gt[i][j] = compute_PJPE(zed_points[j], mocap_points[j])
            smpl_vs_gt[i][j] = compute_PJPE(smpl_points[j], mocap_points[j])
        #     print(
        #         f" Joint {j} (Mocap idx {keys[j]}): ZED vs GT: {zed_vs_gt[i][j]:.5f}, SMPL vs GT: {smpl_vs_gt[i][j]:.5f}"
        #     )

    # remove nan entries
    for i in range(len(zed_vs_gt) - 1, -1, -1):
        if np.any(np.isnan(zed_vs_gt[i])) or np.any(np.isnan(smpl_vs_gt[i])):
            del zed_vs_gt[i]
            del smpl_vs_gt[i]

    NUM_FRAMES = len(zed_vs_gt)
    print(f"Computed errors for {NUM_FRAMES} valid frames")

    all_zed_errors = [
        zed_vs_gt[i][j] for i in range(NUM_FRAMES) for j in range(NUM_JOINTS)
    ]
    all_smpl_errors = [
        smpl_vs_gt[i][j] for i in range(NUM_FRAMES) for j in range(NUM_JOINTS)
    ]
    print(
        f"All errors: ZED vs GT: {np.mean(all_zed_errors):.5f}, SMPL vs GT: {np.mean(all_smpl_errors):.5f}"
    )
    # Compute avg error across all frames
    mpjpe_zed = []
    mpjpe_smpl = []
    for i in range(NUM_FRAMES):
        smpl_frame_errors = [smpl_vs_gt[i][j] for j in range(NUM_JOINTS)]
        mpjpe_smpl.append(np.mean(smpl_frame_errors))
        zed_frame_errors = [zed_vs_gt[i][j] for j in range(NUM_JOINTS)]
        mpjpe_zed.append(np.mean(zed_frame_errors))
    print(
        f"MPJPE ZED vs GT: {np.mean(mpjpe_zed):.5f}, SMPL vs GT: {np.mean(mpjpe_smpl):.5f}"
    )
    # print average per joint
    for j in range(NUM_JOINTS):
        joint_zed_errors = [zed_vs_gt[i][j] for i in range(NUM_FRAMES)]
        joint_smpl_errors = [smpl_vs_gt[i][j] for i in range(NUM_FRAMES)]
        print(
            f"Joint {SMPL_JOINT_NAMES[j]} (Mocap idx {keys[j]}): ZED vs GT: {np.mean(joint_zed_errors):.5f}, SMPL vs GT: {np.mean(joint_smpl_errors):.5f}"
        )


if __name__ == "__main__":
    main()
