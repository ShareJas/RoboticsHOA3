from forward_kinematics import rotation_matrix, homogenous_transformation_matrix
import numpy as np

if __name__ == "__main__":
    # testing code

    # rotation matrix
    axis = np.array([0, 1, 0])
    angle = 2.79
    # matrix = rotation_matrix(axis, angle)
    # print(matrix)

    # homogenous transformation matrix
    v_A = np.array([1, 2, 3])
    print(homogenous_transformation_matrix(axis, angle, v_A))