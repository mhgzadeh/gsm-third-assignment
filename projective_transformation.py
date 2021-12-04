import numpy as np


def transform_points(raw_data, x_cap_proj):
    for j in range(raw_data.shape[0]):
        raw_data[j][0] = np.array(
            [(x_cap_proj[0] * raw_data[j][0] + x_cap_proj[3] * raw_data[j][1] + x_cap_proj[6]) /
             (x_cap_proj[2] * raw_data[j][0] + x_cap_proj[5] * raw_data[j][1] + 1)]
        )
        raw_data[j][0] = np.array(
            [(x_cap_proj[1] * raw_data[j][0] + x_cap_proj[4] * raw_data[j][1] + x_cap_proj[7]) / (
                    x_cap_proj[2] * raw_data[j][0] + x_cap_proj[5] * raw_data[j][1] + 1)]
        )

    return raw_data
