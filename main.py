import numpy as np

import matplotlib.image as mimg
import matplotlib.pyplot as plt

from digitizing import digitizing_point

import os

from projective_coefficients import projective_coefficients
from projective_transformation import transform_points
from quadratic_a_matrix import a_mat_quadratic

if __name__ == "__main__":

    # ************************** Data Prepration ***************************
    gcps = np.array([[5, 82, 320], [1000, 118, 400], [1000, 931, 420],
                     [18, 1000, 390], [194, 249, 303.5], [-23, 721, 422.3],
                     [247, 1119, 445.0], [257, 855, 447.8], [596, 531, 423.9],
                     [770, 362, 405.1]])
    u = 6
    img = mimg.imread('img.jpg')

    # *************** Get 10 Raw points *******************
    if os.path.isfile('raw_data.txt'):
        raw_data = np.loadtxt('raw_data.txt')
    else:
        print('\nUse right click to add point,\n '
              'Backspace to delete worng point and\n '
              'Enter to finish digitizing\n')
        height = np.empty((0, gcps.shape[0]))
        raw_data = digitizing_point(img_mat=img, height_value=height)
        np.savetxt('raw_data.txt', raw_data)
        raw_data = np.loadtxt('raw_data_20.txt')

    # *************** Get 5 Personal Raw points *******************
    if os.path.isfile('personal_raw_data_20.txt'):
        personal_raw_data = np.loadtxt('personal_raw_data_20.txt')
    else:
        print('\nUse right click to add point,\n '
              'Backspace to delete worng point and\n '
              'Enter to finish digitizing\n')
        height = np.array([[320, 400, 420, 370, 430]])
        personal_raw_data = digitizing_point(img_mat=img, height_value=height)
        np.savetxt('personal_raw_data.txt', personal_raw_data)
        personal_raw_data = np.loadtxt('personal_raw_data.txt')

    # *************** Find the Coefficients of Projective Transformation *******************
    x_cap_proj = projective_coefficients(gcp=gcps, raw=raw_data)

    # *************** Create dataset with 15 points *******************
    personal_raw_data_projected = transform_points(personal_raw_data, x_cap_proj)
    full_dataset = np.concatenate((gcps, personal_raw_data_projected), axis=0)

    # *************** RMSE Estimation with 10 and 15 Points *******************
    x_cap_quad = []
    for i in [10, 15]:
        a_mat = a_mat_quadratic(full_dataset[:i, :])
        x_cap = np.linalg.inv(np.transpose(a_mat).dot(a_mat)).dot(np.transpose(a_mat)).dot(full_dataset[:i, 2])

        z_cap = a_mat.dot(x_cap)
        print(f'x_cap of {i} point is equal to:\n {x_cap} \n')
        z = full_dataset[:i, 2]
        v = [z_cap[i] - z[i] for i in range(z_cap.shape[0])]
        rmse = np.sqrt((np.transpose(np.array(v)).dot(np.array(v))) / (i - u))
        print(f'RMSE of {i} point is equal to: {rmse} ')
        print('\n', '*' * 25, '\n')
        x_cap_quad.append(x_cap)

    # *************** Z0 Estimation with 10 and 15 Points *******************
    loi = np.array([[260, 520]])
    z0_10point = a_mat_quadratic(loi).dot(x_cap_quad[0])
    z0_15point = a_mat_quadratic(loi).dot(x_cap_quad[1])
    print(f'z0_10point: {z0_10point[0]},\t z0_15point: {z0_15point[0]}')
    print('\n', '*' * 25, '\n')

    # *************** Plot 3D: Personal Points and GCPs *******************
    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.scatter3D(full_dataset[:10, 0], full_dataset[:10, 1], full_dataset[:10, 2], marker='^', cmap='Reds', s=300,
                 label='GCPs')
    ax.scatter3D(full_dataset[10:15, 0], full_dataset[10:15, 1], full_dataset[10:15, 2], marker='o', cmap='Reds', s=300,
                 label='Personal Points')
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.set_zlabel('$z$', fontsize=30)
    plt.title("Plot 3D: Personal Points and GCPs", fontsize=30)
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=30, bbox_to_anchor=(0, 0))

    # *************** Plot 3D Surface *******************
    plt.figure(2)
    xgrid = np.linspace(-50, 1100, 50)
    ygrid = np.linspace(0, 1200, 50)
    x, y = np.meshgrid(xgrid, ygrid)
    z = np.array([a_mat_quadratic([[x[j][i], y[j][i]]]).dot(x_cap_quad[1]) for j in np.arange(ygrid.shape[0])
                  for i in np.arange(xgrid.shape[0])])
    ax = plt.axes(projection='3d')
    ax.plot_surface(x.reshape(50, 50), y.reshape(50, 50), z.reshape(50, 50), rstride=1, cstride=1,
                    cmap='viridis', edgecolor='none')
    ax.scatter3D(full_dataset[:10, 0], full_dataset[:10, 1], full_dataset[:10, 2], marker='^', cmap='Reds', s=300,
                 label='GCPs')
    ax.scatter3D(full_dataset[10:15, 0], full_dataset[10:15, 1], full_dataset[10:15, 2], marker='o', cmap='Reds', s=300,
                 label='Personal Points')
    ax.set_xlabel('$x$', fontsize=30)
    ax.set_ylabel('$y$', fontsize=30)
    ax.set_zlabel('$z$', fontsize=30)
    ax.set_title('Plot 3D Surface, Quadratic Polynomial', fontsize=30)
    plt.legend(loc='upper left', numpoints=1, ncol=3, fontsize=30, bbox_to_anchor=(0, 0))

    # *************** Height Profile *******************
    plt.figure(3)
    x_height = np.linspace(5, 770, 120)
    y_height = np.linspace(82, 362, 120)
    z_height = []
    dist = []
    d = np.sqrt((x_height[0] - x_height[1]) ** 2 + (y_height[0] - y_height[1]) ** 2)
    z_height = np.empty((0, 1))
    dist = np.empty((0, 1))
    for i in np.arange(1, x_height.shape[0]):
        z_height = np.append(z_height, a_mat_quadratic([[x_height[i], y_height[i]]]).dot(x_cap_quad[1]))
        dist = np.append(dist, np.array([d * i]))
    plt.plot(dist, z_height, 'b-', dist[0], z_height[0], 'ro', dist[-1], z_height[-1], 'go', markersize=10)
    plt.xlabel('Distance')
    plt.ylabel('Height')
    plt.title('Height Profile', fontsize=30)
    plt.legend(['Height Profile', 'Corner 1', 'Tor'])
    plt.show()
