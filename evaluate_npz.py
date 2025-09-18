import numpy as np
from scipy.spatial import procrustes
import argparse


# ---- Load Data from NPZ ----
def load_data_from_npz(npz_path):
    """
    Load the synchronized data from an NPZ file.
    """
    data = np.load(npz_path, allow_pickle=True)
    print(f"Loaded fields: {list(data.keys())}")
    return data


# ---- Procrustes Alignment ----
def align_skeletons(zed_joints, smpl_joints):
    """
    Align ZED skeleton to SMPL skeleton using Procrustes Analysis.

    zed_joints: (N, 3) numpy array of ZED joint positions
    smpl_joints: (N, 3) numpy array of SMPL joint positions

    Returns:
    aligned_zed_joints: (N, 3) numpy array of aligned ZED joints
    transformation: dictionary with R (rotation matrix), s (scale), and t (translation)
    """
    # Perform Procrustes analysis to align ZED joints to SMPL joints
    mtx1, mtx2, disparity = procrustes(smpl_joints, zed_joints)

    # Extract the transformation parameters: R (rotation matrix), s (scale), t (translation)
    R = mtx2.T @ mtx1  # Rotation matrix
    scale = np.linalg.norm(mtx1) / np.linalg.norm(mtx2)  # Scaling factor
    t = np.mean(mtx1, axis=0) - scale * np.mean(mtx2, axis=0)  # Translation vector

    # Apply the transformation to ZED joints
    aligned_zed_joints = scale * mtx2 @ R.T + t

    return aligned_zed_joints, {"R": R, "s": scale, "t": t}


# ---- Compute Joint Error ----
def compute_joint_error(aligned_zed_joints, smpl_joints):
    """
    Compute the joint error as the average Euclidean distance between aligned ZED and SMPL joints.

    aligned_zed_joints: (N, 3) numpy array of aligned ZED joint positions
    smpl_joints: (N, 3) numpy array of SMPL joint positions

    Returns:
    error: float, the average Euclidean distance between the aligned joints
    """
    errors = np.linalg.norm(aligned_zed_joints - smpl_joints, axis=1)
    average_error = np.mean(errors)
    return average_error


# ---- Process Skeleton Data ----
def process_skeleton_data(data):
    """
    Process the skeleton data for alignment and error calculation.
    """
    for i, (mocap_markers, smpl_keypoints) in enumerate(
        zip(data["markers"], data["smpl_keypoints"])
    ):
        # Ensure keypoints are in the correct shape (N, 3)
        mocap_markers = np.array(mocap_markers)
        smpl_keypoints = np.array(smpl_keypoints)

        # Check if the dimensions are consistent
        if mocap_markers.shape != smpl_keypoints.shape:
            print(
                f"Skipping frame {i}: Inconsistent joint count: "
                f"mocap {mocap_markers.shape}, smpl {smpl_keypoints.shape}"
            )
            continue

        # Align ZED skeleton to SMPL skeleton
        aligned_mocap_markers, transformation = align_skeletons(
            mocap_markers, smpl_keypoints
        )

        # Compute joint error after alignment
        joint_error = compute_joint_error(aligned_mocap_markers, smpl_keypoints)

        # Output the result
        print(f"Frame {i}: Joint Error = {joint_error:.4f}")


# ---- Main Function ----
def main(npz_path):
    """
    Main function to process the skeleton data from an NPZ file.
    """
    # Load the data from the NPZ file
    data = load_data_from_npz(npz_path)

    # Process the skeleton data (align and compute joint errors)
    process_skeleton_data(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate skeleton data from NPZ file")
    parser.add_argument(
        "--npz_path", type=str, required=True, help="Path to the NPZ file"
    )
    args = parser.parse_args()
    main(args.npz_path)
