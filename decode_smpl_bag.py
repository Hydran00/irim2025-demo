#!/usr/bin/env python3

import argparse
import numpy as np
from tqdm import tqdm

from dataclasses import dataclass
from typing import List
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

from scipy.spatial import procrustes
from sensor_msgs_py import point_cloud2 as pc2
from utils import pointcloud2_to_xyz_array
from visualize import visualize_synced_data


# ---- Data structures ----
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
class SyncedData:
    timestamp: int
    cloud: object
    mocap_skeleton: List[MarkerData]
    smpl: SMPLData


# ---- ROS 2 bag reader ----
def read_bag(bag_path: str):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msgs_by_topic = {"/human_cloud": [], "/skeletons": [], "/smpl_params": []}

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic in msgs_by_topic:
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)
            msgs_by_topic[topic].append((t, msg))

    return msgs_by_topic


# ---- Sync messages ----
def sync_messages(msgs_by_topic, slop_ns=5e7):
    def ros_time_to_ns(stamp):
        return stamp.sec * 1_000_000_000 + stamp.nanosec

    clouds = msgs_by_topic["/human_cloud"]
    mocap_skeletons = msgs_by_topic["/skeletons"]
    smpls = msgs_by_topic["/smpl_params"]

    skeleton_by_time = {ros_time_to_ns(msg.header.stamp): msg for _, msg in mocap_skeletons}
    smpl_by_time = {ros_time_to_ns(msg.header.stamp): msg for _, msg in smpls}

    synced = []
    for t_cloud, cloud in clouds:
        if not skeleton_by_time or not smpl_by_time:
            continue

        t_skel_ns, skel_msg = min(skeleton_by_time.items(), key=lambda kv: abs(kv[0] - t_cloud))
        t_smpl_ns, smpl_msg = min(smpl_by_time.items(), key=lambda kv: abs(kv[0] - t_cloud))

        if abs(t_skel_ns - t_cloud) < slop_ns and abs(t_smpl_ns - t_cloud) < slop_ns:
            mocap_marker_list = [
                MarkerData(
                    id_type=0,
                    marker_index=0,
                    marker_name=skel.skeleton_name,
                    translation=np.array([rb.pose.position.x, rb.pose.position.y, rb.pose.position.z]),
                )
                for skel in skel_msg.skeletons
                for rb in skel.rigid_bodies
            ]

            smpl = SMPLData(
                body_pose=np.array(smpl_msg.body_pose),
                transl=np.array(smpl_msg.transl),
                global_orient=np.array(smpl_msg.global_orient),
                betas=np.array(smpl_msg.betas),
                keypoints=np.array(smpl_msg.keypoints).reshape(-1, 3),
            )

            synced.append(SyncedData(timestamp=t_cloud, cloud=cloud, mocap_skeleton=mocap_marker_list, smpl=smpl))

    return synced





# ---- Save NPZ ----
def save_synced_data_to_npz(synced_data, npz_path):
    arrays = {
        "cloud": [],
        "mocap_skeleton": [],
        "smpl_keypoints": [],
        "smpl_betas": [],
        "smpl_body_pose": [],
        "smpl_transl": [],
        "smpl_global_orient": [],
    }

    for entry in tqdm(synced_data, desc="Saving data to npz"):
        arrays["cloud"].append(pointcloud2_to_xyz_array(entry.cloud))
        arrays["mocap_skeleton"].append([m.translation for m in entry.mocap_skeleton])
        arrays["smpl_keypoints"].append(entry.smpl.keypoints)
        arrays["smpl_betas"].append(entry.smpl.betas)
        arrays["smpl_body_pose"].append(entry.smpl.body_pose)
        arrays["smpl_transl"].append(entry.smpl.transl)
        arrays["smpl_global_orient"].append(entry.smpl.global_orient)

    np.savez(npz_path, **{k: np.array(v, dtype=object) for k, v in arrays.items()})
    print(f"Data saved to {npz_path}")


# ---- Main ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ROS 2 bag files and visualize with Open3D")
    parser.add_argument("--bag_path", type=str, required=True, help="Path to the ROS 2 bag file")
    parser.add_argument("--npz_path", type=str, default="", help="Path to save the synchronized data as .npz file")
    args = parser.parse_args()

    msgs_by_topic = read_bag(args.bag_path)
    synced_data = sync_messages(msgs_by_topic)

    print(f"Extracted {len(synced_data)} synchronized entries")

    if args.npz_path:
        save_synced_data_to_npz(synced_data, args.npz_path)
    else:
        visualize_synced_data(synced_data)
