import numpy as np

import matplotlib.pyplot as plt


def digitizing_point(img_mat, height_value, num=-1):
    plt.imshow(img_mat)
    data = np.array(plt.ginput(num, 0, True, 3, 2, None))
    data = np.concatenate((data, height_value.T), axis=1)
    return data
