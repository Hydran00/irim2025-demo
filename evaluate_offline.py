#!/usr/bin/env python3
"""
analyze_from_jsons.py

Compute MPJPE (aligned and unaligned) between:
  - SMPL JSON vs Mocap
  - Optimized JSON joints vs Mocap
  - ZED vs Mocap

Also divide joints into head, torso, and limbs for per-part analysis.
Compute Chamfer Distance between SMPL JSON mesh and scan.
Visualize all skeletons with same color per joint.
"""

import argparse
import json
import os
import numpy as np
from scipy.linalg import orthogonal_procrustes
import open3d as o3d
from convert import SMPL_TO_MOCAP
from scipy.spatial import cKDTree

VISUALIZE = False

# Define joint groups (indices from SMPL_TO_MOCAP keys)
HEAD_JOINTS = [15, 16]  # example: head and neck
TORSO_JOINTS = [0, 1, 2, 3, 8]  # pelvis and spine joints
LIMB_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20]  # arms and legs

JOINT_GROUPS = {"head": HEAD_JOINTS, "torso": TORSO_JOINTS, "limbs": LIMB_JOINTS}


def chamfer_distance(P, Q):
    tree_Q = cKDTree(Q)
    tree_P = cKDTree(P)
    dist_P_to_Q, _ = tree_Q.query(P)
    dist_Q_to_P, _ = tree_P.query(Q)
    cd = np.mean(dist_P_to_Q**2) + np.mean(dist_Q_to_P**2)
    return cd


def create_sphere_marker(center, radius=0.02, color=[1, 0, 0]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh.translate(center)
    mesh.paint_uniform_color(color)
    return mesh


def visualize_skeletons_joint_colors(
    mocap_pts=None,
    smpl_json_pts=None,
    optimised_pts=None,
    zed_pts=None,
    joint_radius=0.02,
    title="Skeletons Comparison",
):
    n_joints = 0
    for pts in [mocap_pts, smpl_json_pts, optimised_pts, zed_pts]:
        if pts is not None:
            n_joints = max(n_joints, pts.shape[0])
    if n_joints == 0:
        return

    rng = np.random.default_rng(42)
    joint_colors = rng.uniform(0.2, 1.0, size=(n_joints, 3))
    geometries = []

    skeletons = {
        "Mocap": mocap_pts,
        "SMPL_JSON": smpl_json_pts,
        "Optimised": optimised_pts,
        "ZED": zed_pts,
    }

    for pts in skeletons.values():
        if pts is None:
            continue
        for idx, pt in enumerate(pts):
            color = joint_colors[idx]
            geometries.append(
                create_sphere_marker(pt, color=color, radius=joint_radius)
            )

    o3d.visualization.draw_geometries(geometries, window_name=title)


def procrustes_rigid_pelvis(A, B, pelvis_idx=0):
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.shape != B.shape:
        raise ValueError("Shapes must match")
    A0, B0 = A - A[pelvis_idx], B - B[pelvis_idx]
    R, _ = orthogonal_procrustes(B0, A0)
    return B0 @ R + A[pelvis_idx]


def load_json_keys(json_path):
    with open(json_path, "r") as f:
        j = json.load(f)

    def safe_get(k):
        return np.array(j[k], dtype=np.float64) if k in j else None

    return {
        "mocap": safe_get("mocap_keypoints"),
        "zed": safe_get("zed_keypoints"),
        "smpl": safe_get("smpl_keypoints"),
    }


def load_optimised_joints(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        j = json.load(f)
    return np.array(j["joints"], dtype=np.float64) if "joints" in j else None


def load_scan_ply(ply_path):
    if not os.path.exists(ply_path):
        return None
    pc = o3d.io.read_point_cloud(ply_path)
    if pc.is_empty():
        return None
    return np.asarray(pc.points, dtype=np.float64)


def subset_to_mapping(smpl_pts, zed_pts, mocap_pts):
    smpl_indices = list(SMPL_TO_MOCAP.keys())
    mocap_indices = [SMPL_TO_MOCAP[k] for k in smpl_indices]

    smpl_pts, mocap_pts = np.asarray(smpl_pts), np.asarray(mocap_pts)
    zed_pts = np.asarray(zed_pts) if zed_pts is not None else None

    smpl_subset = smpl_pts[smpl_indices]
    mocap_subset = mocap_pts[mocap_indices]
    zed_subset = zed_pts[smpl_indices] if zed_pts is not None else None
    return smpl_subset, zed_subset, mocap_subset


def compute_errors(smpl_json_subset, optimised_subset, zed_subset, mocap_subset, pelvis_idx=0):
    smpl_json_al = procrustes_rigid_pelvis(mocap_subset, smpl_json_subset, pelvis_idx)
    optimised_al = (
        procrustes_rigid_pelvis(mocap_subset, optimised_subset, pelvis_idx)
        if optimised_subset is not None
        else None
    )
    zed_al = (
        procrustes_rigid_pelvis(mocap_subset, zed_subset, pelvis_idx)
        if zed_subset is not None
        else None
    )

    return {
        "aligned": {"smpl_json": smpl_json_al, "optimised": optimised_al, "zed": zed_al},
        "unaligned": {
            "smpl_json": smpl_json_subset,
            "optimised": optimised_subset,
            "zed": zed_subset,
        },
    }


def compute_metrics(errors_dict, mocap_subset):
    metrics = {}
    for key, pts in errors_dict.items():
        if pts is None:
            metrics[key] = None
            continue
        diff = np.linalg.norm(pts - mocap_subset, axis=1)
        per_group = {}
        for group, indices in JOINT_GROUPS.items():
            per_group[group] = diff[indices].mean()
        metrics[key] = {"overall": diff.mean(), "groups": per_group}
    return metrics


def find_param_jsons(folder):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith("_params.json") and f.startswith("cloud_")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--pelvis_idx", type=int, default=0)
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    json_files = find_param_jsons(args.folder)
    if not json_files:
        raise SystemExit("No JSON files found.")

    accum_metrics = {"aligned": [], "unaligned": []}
    chamfers = []
    valid_frames = 0

    for json_path in json_files:
        stem = os.path.splitext(os.path.basename(json_path))[0].replace("_params", "")
        cloud_path = os.path.join(args.folder, f"{stem}.ply")
        cloud_pts = load_scan_ply(cloud_path)

        data = load_json_keys(json_path)
        mocap, zed, smpl_json = data["mocap"], data["zed"], data["smpl"]

        optimised_path = os.path.join(args.folder, f"{stem}_optimised_params.json")
        optimised = load_optimised_joints(optimised_path)

        if mocap is None or smpl_json is None or optimised is None:
            continue

        try:
            smpl_json_sub, zed_sub, mocap_sub = subset_to_mapping(smpl_json, zed, mocap)
            optimised_sub, _, _ = subset_to_mapping(optimised, zed, mocap)
        except Exception as e:
            print(f"[WARN] {stem} subset failed: {e}")
            continue

        errors = compute_errors(
            smpl_json_sub, optimised_sub, zed_sub, mocap_sub, pelvis_idx=args.pelvis_idx
        )

        frame_metrics_aligned = compute_metrics(errors["aligned"], mocap_sub)
        frame_metrics_unaligned = compute_metrics(errors["unaligned"], mocap_sub)
        accum_metrics["aligned"].append(frame_metrics_aligned)
        accum_metrics["unaligned"].append(frame_metrics_unaligned)

        if cloud_pts is not None:
            cd_json = chamfer_distance(smpl_json, cloud_pts)
            print(f"[Frame {stem}] Chamfer(JSON vs Scan) = {cd_json:.6f}")
            cd_opt = chamfer_distance(optimised, cloud_pts)
            print(f"[Frame {stem}] Chamfer(Optimised vs Scan) = {cd_opt:.6f}")
            chamfers.append((cd_json, cd_opt))

        valid_frames += 1

        if VISUALIZE and cloud_pts is not None:
            visualize_skeletons_joint_colors(
                mocap_pts=mocap_sub,
                smpl_json_pts=smpl_json_sub,
                optimised_pts=optimised_sub,
                zed_pts=zed_sub,
            )

    if valid_frames == 0:
        raise SystemExit("No valid frames processed.")

    def summarize(accum_list, label):
        print(f"\n=== {label} Metrics ===")
        keys = ["smpl_json", "optimised", "zed"]
        for key in keys:
            all_overall = [f[key]["overall"] for f in accum_list if f[key] is not None]
            if not all_overall:
                continue
            print(
                f"{key}: mean={np.mean(all_overall):.4f}, std={np.std(all_overall):.4f}"
            )
            for group in JOINT_GROUPS.keys():
                all_group = [
                    f[key]["groups"][group] for f in accum_list if f[key] is not None
                ]
                print(
                    f"  {group}: mean={np.mean(all_group):.4f}, std={np.std(all_group):.4f}"
                )

    summarize(accum_metrics["unaligned"], "Unaligned")
    summarize(accum_metrics["aligned"], "Aligned")

    if chamfers:
        chamfers = np.array(chamfers)
        print(
            f"\nChamfer(JSON vs Scan): mean={chamfers[:,0].mean():.6f}, std={chamfers[:,0].std():.6f}"
        )
        print(
            f"Chamfer(Optimised vs Scan): mean={chamfers[:,1].mean():.6f}, std={chamfers[:,1].std():.6f}"
        )

    print(f"\nProcessed {valid_frames} frames. Done.")


if __name__ == "__main__":
    main()
