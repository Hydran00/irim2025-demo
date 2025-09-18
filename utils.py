
from sensor_msgs_py import point_cloud2 as pc2
import numpy as np
# ---- PointCloud2 helper ----
def pointcloud2_to_xyz_array(cloud_msg):
    return np.array(
        [[p[0], p[1], p[2]] for p in pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)],
        dtype=np.float32,
    )