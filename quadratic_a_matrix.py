import numpy as np


def a_mat_quadratic(data):
    a_mat = np.empty((0, 6))
    for point in data:
        a_mat = np.append(
            a_mat,
            np.array([[1, point[0], point[1], point[0]**2, point[1]**2, point[0]*point[1]]]),
            axis=0
        )
    return a_mat
