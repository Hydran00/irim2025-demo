#!/usr/bin/env python3

import argparse
import os
import numpy as np
from dataclasses import dataclass
from typing import List

import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from tqdm import tqdm
import trimesh

from sensor_msgs_py import point_cloud2 as pc2
from utils import pointcloud2_to_xyzrgb_array
import json

# your helper


# ---------------- Data structures ----------------
@dataclass
class MarkerData:
    id_type: int
    marker_index: int
    marker_name: str
    translation: np.ndarray


@dataclass
class SMPLData:
    body_pose: np.ndarray
    transl: np.ndarray
    global_orient: np.ndarray
    betas: np.ndarray
    keypoints: np.ndarray


@dataclass
class ZEDSkeleton:
    timestamp: int
    keypoints: np.ndarray


@dataclass
class SyncedData:
    timestamp: int
    cloud: object
    mocap_skeleton: List[MarkerData]
    smpl: SMPLData
    zed_skeleton: ZEDSkeleton


# ---------------- ROS 2 bag reader ----------------
def read_bag(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msgs_by_topic = {
        "/human_cloud": [],
        "/skeletons": [],
        "/smpl_params": [],
        "/smpl_keypoints": [],
    }

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in msgs_by_topic:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            msgs_by_topic[topic].append((t, msg))

    return msgs_by_topic


# ---------------- Helper to convert ROS time ----------------
def ros_time_to_ns(stamp):
    return stamp.sec * 1_000_000_000 + stamp.nanosec


# ---------------- Estimate safe timestamp slop ----------------
def estimate_slop_ns(clouds, mocap_skeletons, smpls, zed_keypoints_msgs):
    cloud_times = [ros_time_to_ns(c.header.stamp) for _, c in clouds]
    mocap_times = [ros_time_to_ns(m.header.stamp) for _, m in mocap_skeletons]
    smpl_times = [ros_time_to_ns(s.header.stamp) for _, s in smpls]
    zed_times = [ros_time_to_ns(z.header.stamp) for _, z in zed_keypoints_msgs]

    deltas = []
    for t_cloud in cloud_times:
        nearest_mocap = min(mocap_times, key=lambda t: abs(t - t_cloud))
        nearest_smpl = min(smpl_times, key=lambda t: abs(t - t_cloud))
        nearest_zed = min(zed_times, key=lambda t: abs(t - t_cloud))
        deltas.append(abs(nearest_mocap - t_cloud))
        deltas.append(abs(nearest_smpl - t_cloud))
        deltas.append(abs(nearest_zed - t_cloud))

    slop_ns = int(np.percentile(deltas, 95))
    print(f"Estimated slop_ns: {slop_ns} ns (~{slop_ns/1e6:.1f} ms)")
    return slop_ns


# ---------------- Sync messages ----------------
def sync_messages(msgs_by_topic):
    clouds = [
        (ros_time_to_ns(cloud.header.stamp), cloud)
        for _, cloud in msgs_by_topic["/human_cloud"]
    ]
    mocap_skeletons = msgs_by_topic["/skeletons"]
    smpls = msgs_by_topic["/smpl_params"]
    smpl_keypoints_msgs = msgs_by_topic["/smpl_keypoints"]

    slop_ns = estimate_slop_ns(clouds, mocap_skeletons, smpls, smpl_keypoints_msgs)

    skeleton_by_time = {
        ros_time_to_ns(msg.header.stamp): msg for _, msg in mocap_skeletons
    }
    smpl_by_time = {ros_time_to_ns(msg.header.stamp): msg for _, msg in smpls}
    smpl_keypoints_by_time = {
        ros_time_to_ns(msg.header.stamp): msg for _, msg in smpl_keypoints_msgs
    }

    synced = []

    for t_cloud, cloud in tqdm(clouds, desc="Synchronizing messages"):
        if not skeleton_by_time or not smpl_by_time or not smpl_keypoints_by_time:
            continue

        t_skel_ns, skel_msg = min(
            skeleton_by_time.items(), key=lambda kv: abs(kv[0] - t_cloud)
        )
        t_smpl_ns, smpl_msg = min(
            smpl_by_time.items(), key=lambda kv: abs(kv[0] - t_cloud)
        )
        t_smpl_keypoints_ns, smpl_keypoints_msg = min(
            smpl_keypoints_by_time.items(), key=lambda kv: abs(kv[0] - t_cloud)
        )

        if (
            max(
                abs(t_skel_ns - t_cloud),
                abs(t_smpl_ns - t_cloud),
                abs(t_smpl_keypoints_ns - t_cloud),
            )
            < slop_ns
        ):

            mocap_marker_list = [
                MarkerData(
                    id_type=0,
                    marker_index=0,
                    marker_name=skel.skeleton_name,
                    translation=np.array(
                        [rb.pose.position.x, rb.pose.position.y, rb.pose.position.z]
                    ),
                )
                for skel in skel_msg.skeletons
                for rb in skel.rigid_bodies
            ]

            zed_keypoints = np.array(smpl_msg.keypoints).reshape(-1, 3)
            smpl_keypoints = np.array(
                [
                    [pose.position.x, pose.position.y, pose.position.z]
                    for pose in smpl_keypoints_msg.poses
                ]
            ).reshape(-1, 3)

            smpl = SMPLData(
                body_pose=np.array(smpl_msg.body_pose),
                transl=np.array(smpl_msg.transl),
                global_orient=np.array(smpl_msg.global_orient),
                betas=np.array(smpl_msg.betas),
                keypoints=smpl_keypoints,
            )
            zed_keypoints_data = ZEDSkeleton(
                timestamp=t_smpl_keypoints_ns, keypoints=zed_keypoints
            )

            synced.append(
                SyncedData(
                    timestamp=t_cloud,
                    cloud=cloud,
                    mocap_skeleton=mocap_marker_list,
                    smpl=smpl,
                    zed_skeleton=zed_keypoints_data,
                )
            )

    print(f"Synchronized frames: {len(synced)}")
    return synced


# ---------------- Save point clouds individually ----------------
# ---------------- Save point clouds individually ----------------
def save_pointclouds(synced_data, folder_path):
    os.makedirs(folder_path, exist_ok=True)
    prev_cloud_array = None

    for i, entry in enumerate(tqdm(synced_data, desc="Saving point clouds")):
        cloud_array = pointcloud2_to_xyzrgb_array(entry.cloud)

        # Separate xyz and rgb
        xyz = cloud_array[:, :3]
        rgb = cloud_array[:, 3:6].astype(np.uint8)  # ensure uint8 for colors

        # Save only if changed
        if prev_cloud_array is None or not np.array_equal(
            prev_cloud_array, cloud_array
        ):
            # Save PLY
            mesh = trimesh.PointCloud(vertices=xyz, colors=rgb)
            ply_filename = os.path.join(folder_path, f"cloud_{i:05d}.ply")
            mesh.export(ply_filename)
            prev_cloud_array = cloud_array.copy()

            # Save JSON with SMPL + ZED + Mocap keypoints
            json_filename = os.path.join(folder_path, f"cloud_{i:05d}_params.json")
            json_data = {
                "smpl_body_pose": entry.smpl.body_pose.tolist(),
                "smpl_transl": entry.smpl.transl.tolist(),
                "smpl_global_orient": entry.smpl.global_orient.tolist(),
                "smpl_betas": entry.smpl.betas.tolist(),
                "smpl_keypoints": entry.smpl.keypoints.tolist(),
                "zed_keypoints": entry.zed_skeleton.keypoints.tolist(),
                "mocap_keypoints": [
                    m.translation.tolist() for m in entry.mocap_skeleton
                ],
            }
            with open(json_filename, "w") as f:
                json.dump(json_data, f, indent=2)


# ---------------- Save synced keypoints & SMPL as NPZ ----------------
def save_synced_data_to_npz(synced_data, npz_path):
    arrays = {
        "mocap_skeleton": [],
        "smpl_keypoints": [],
        "zed_keypoints": [],
        "smpl_betas": [],
        "smpl_body_pose": [],
        "smpl_transl": [],
        "smpl_global_orient": [],
    }

    for entry in tqdm(synced_data, desc="Saving keypoints to npz"):
        arrays["mocap_skeleton"].append([m.translation for m in entry.mocap_skeleton])
        arrays["smpl_keypoints"].append(entry.smpl.keypoints)
        arrays["zed_keypoints"].append(entry.zed_skeleton.keypoints)
        arrays["smpl_betas"].append(entry.smpl.betas)
        arrays["smpl_body_pose"].append(entry.smpl.body_pose)
        arrays["smpl_transl"].append(entry.smpl.transl)
        arrays["smpl_global_orient"].append(entry.smpl.global_orient)

    np.savez(npz_path, **{k: np.array(v, dtype=object) for k, v in arrays.items()})
    print(f"Data saved to {npz_path}")


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ROS 2 bag and save point clouds & keypoints"
    )
    parser.add_argument(
        "--bag_path", type=str, required=True, help="Path to the ROS 2 bag file"
    )
    parser.add_argument(
        "--npz_path",
        type=str,
        default="",
        help="Path to save synchronized keypoints as .npz",
    )
    parser.add_argument(
        "--cloud_folder", type=str, default="clouds", help="Folder to save point clouds"
    )
    args = parser.parse_args()

    msgs_by_topic = read_bag(args.bag_path)
    print("Finished reading bag")

    synced_data = sync_messages(msgs_by_topic)
    print(f"Extracted {len(synced_data)} synchronized entries")

    if args.npz_path:
        save_synced_data_to_npz(synced_data, args.npz_path)
    if args.cloud_folder:
        save_pointclouds(synced_data, args.cloud_folder)
