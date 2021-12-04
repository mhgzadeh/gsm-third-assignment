import numpy as np


def projective_coefficients(gcp, raw):
    a_mat = np.empty((0, 8))
    x_y = np.empty((0, 1))
    for i in range(8):
        a_mat = np.append(
            a_mat,
            np.array([[raw[i][0], 0, -1 * raw[i][0] * gcp[i][0], raw[i][1], 0, -1 * raw[i][1] * gcp[i][0], 1, 0]]),
            axis=0
        )
        a_mat = np.append(
            a_mat,
            np.array([[0, raw[i][0], -1 * raw[i][0] * gcp[i][1], 0, raw[i][1], -1 * raw[i][1] * gcp[i][1], 0, 1]]),
            axis=0
        )
        x_y = np.append(x_y, np.array(gcp[i][0]))
        x_y = np.append(x_y, np.array(gcp[i][1]))
    x_cap = np.linalg.inv(np.transpose(a_mat).dot(a_mat)).dot(np.transpose(a_mat)).dot(x_y)
    z_check = np.empty((0, 1))
    x_y_check = np.empty((0, 1))
    for i in range(2):
        j = 8 + i
        z_check = np.append(
            z_check,
            np.array([(x_cap[0] * raw[j][0] + x_cap[3] * raw[j][1] + x_cap[6]) / (
                        x_cap[2] * raw[j][0] + x_cap[5] * raw[j][1] + 1)])
        )
        z_check = np.append(
            z_check,
            np.array([(x_cap[1] * raw[j][0] + x_cap[4] * raw[j][1] + x_cap[7]) / (
                        x_cap[2] * raw[j][0] + x_cap[5] * raw[j][1] + 1)])
        )
        x_y_check = np.append(x_y_check, np.array(gcp[j][0]))
        x_y_check = np.append(x_y_check, np.array(gcp[j][1]))
    for i in range(4):
        rmse = z_check[i] - x_y_check[i]
        return x_cap
