from sensor_msgs_py import point_cloud2 as pc2
import numpy as np

import numpy as np
import sensor_msgs_py.point_cloud2 as pc2


def pointcloud2_to_xyzrgb_array(cloud_msg):
    """
    Convert a sensor_msgs/PointCloud2 into a Nx6 numpy array.
    Handles both 'rgb' packed float32 field and separate 'r','g','b' fields.
    """
    # Get field names present in the cloud
    field_names = [f.name for f in cloud_msg.fields]

    points = []
    if set(["r", "g", "b"]).issubset(field_names):
        # Case 1: Separate r,g,b fields exist
        for p in pc2.read_points(
            cloud_msg, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=True
        ):
            points.append([p[0], p[1], p[2], p[3], p[4], p[5]])

    elif "rgb" in field_names or "rgba" in field_names:
        # Case 2: Packed rgb/rgba float32 field
        color_field = "rgb" if "rgb" in field_names else "rgba"
        for p in pc2.read_points(
            cloud_msg, field_names=("x", "y", "z", color_field), skip_nans=True
        ):
            x, y, z, rgb = p
            # unpack float32 to int
            rgb_int = np.frombuffer(np.float32(rgb).tobytes(), dtype=np.uint32)[0]
            r = (rgb_int >> 16) & 255
            g = (rgb_int >> 8) & 255
            b = rgb_int & 255
            points.append([x, y, z, r, g, b])
    else:
        raise ValueError(
            f"PointCloud2 has no rgb or r,g,b fields. Fields: {field_names}"
        )

    return np.array(points, dtype=np.float32)
