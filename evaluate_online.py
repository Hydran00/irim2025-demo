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
    """Align B to A using rigid Procrustes with pelvis as pivot."""
    A0 = A - A[pelvis_idx]
    B0 = B - B[pelvis_idx]
    R, _ = orthogonal_procrustes(B0, A0)
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

    ax.scatter(mocap_points[:, 0], mocap_points[:, 1], mocap_points[:, 2], c="g", label="Mocap")
    if smpl_points is not None:
        ax.scatter(smpl_points[:, 0], smpl_points[:, 1], smpl_points[:, 2], c="orange", label="SMPL")
    if zed_points is not None:
        ax.scatter(zed_points[:, 0], zed_points[:, 1], zed_points[:, 2], c="b", label="ZED")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Mocap (Green), SMPL (Orange), ZED (Blue)")
    ax.legend()

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
        plot_skeleton(mocap_subset, smpl_points=smpl_subset, zed_points=zed_subset)

    smpl_aligned = procrustes_rigid_pelvis(mocap_subset, smpl_subset)
    zed_aligned = procrustes_rigid_pelvis(mocap_subset, zed_subset)

    if visualize:
        plot_skeleton(mocap_subset, smpl_points=smpl_aligned)
        plot_skeleton(mocap_subset, zed_points=zed_aligned)
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
    zed_vs_gt = [[np.nan for _ in range(NUM_JOINTS)] for _ in range(NUM_FRAMES)]
    smpl_vs_gt = [[np.nan for _ in range(NUM_JOINTS)] for _ in range(NUM_FRAMES)]
    keys = list(SMPL_TO_MOCAP.keys())

    for i in range(NUM_FRAMES):
        print(f"Processing frame {i}")
        smpl_points, zed_points, mocap_points = visualize_npz(
            data, frame_idx=i + args.frame, visualize=VISUALIZE
        )
        if smpl_points is None or zed_points is None or mocap_points is None:
            continue
        for j in range(NUM_JOINTS):
            zed_vs_gt[i][j] = compute_PJPE(zed_points[j], mocap_points[j])
            smpl_vs_gt[i][j] = compute_PJPE(smpl_points[j], mocap_points[j])

    # Remove invalid frames
    for i in range(len(zed_vs_gt) - 1, -1, -1):
        if np.any(np.isnan(zed_vs_gt[i])) or np.any(np.isnan(smpl_vs_gt[i])):
            del zed_vs_gt[i]
            del smpl_vs_gt[i]

    NUM_FRAMES = len(zed_vs_gt)
    print(f"Computed errors for {NUM_FRAMES} valid frames")

    # Flatten all errors
    all_zed_errors = [zed_vs_gt[i][j] for i in range(NUM_FRAMES) for j in range(NUM_JOINTS)]
    all_smpl_errors = [smpl_vs_gt[i][j] for i in range(NUM_FRAMES) for j in range(NUM_JOINTS)]
    print(f"All errors: ZED vs GT: {np.mean(all_zed_errors):.5f}, SMPL vs GT: {np.mean(all_smpl_errors):.5f}")

    # Frame-level MPJPE
    mpjpe_zed = [np.mean(zed_vs_gt[i]) for i in range(NUM_FRAMES)]
    mpjpe_smpl = [np.mean(smpl_vs_gt[i]) for i in range(NUM_FRAMES)]
    print(f"MPJPE ZED vs GT: {np.mean(mpjpe_zed):.5f}, SMPL vs GT: {np.mean(mpjpe_smpl):.5f}")

    # Per-joint averages
    for j in range(NUM_JOINTS):
        joint_zed_errors = [zed_vs_gt[i][j] for i in range(NUM_FRAMES)]
        joint_smpl_errors = [smpl_vs_gt[i][j] for i in range(NUM_FRAMES)]
        print(
            f"Joint {SMPL_JOINT_NAMES[j]} (Mocap idx {keys[j]}): "
            f"ZED vs GT: {np.mean(joint_zed_errors):.5f}, SMPL vs GT: {np.mean(joint_smpl_errors):.5f}"
        )

    # ---------------- Region-level averages ----------------
    REGION_MAP = {
        "torso": [0, 1, 2, 3, 8, 12, 13],  # pelvis, spine1, spine3, neck
        "head": [11, 14],            # head
        "legs": [4, 5, 6, 7, 9, 10],  # hips, knees, ankles, feet
        "arms": [15, 16, 17, 18, 19, 20],  # collars, shoulders, elbows, wrists
    }

    print("\n--- Region-level MPJPE ---")
    for region, idxs in REGION_MAP.items():
        zed_region_errors = [zed_vs_gt[i][j] for i in range(NUM_FRAMES) for j in idxs]
        smpl_region_errors = [smpl_vs_gt[i][j] for i in range(NUM_FRAMES) for j in idxs]
        print(
            f"{region.capitalize()} → ZED vs GT: {np.mean(zed_region_errors):.5f}, "
            f"SMPL vs GT: {np.mean(smpl_region_errors):.5f}"
        )


if __name__ == "__main__":
    main()
