import math
import numpy as np
import copy
from reacher import forward_kinematics

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2
TOLERANCE = 0.01 # tolerance for inverse kinematics
PERTURBATION = 0.0001 # perturbation for finite difference method
MAX_ITERATIONS = 1

def ik_cost(end_effector_pos, guess):
    """Calculates the inverse kinematics cost.

    This function computes the inverse kinematics cost, which represents the Euclidean
    distance between the desired end-effector position and the end-effector position
    resulting from the provided 'guess' joint angles.

    Args:
        end_effector_pos (numpy.ndarray), (3,): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray), (3,): A guess at the joint angles to achieve the desired end-effector
            position. A numpy array with 3 elements.

    Returns:
        float: The Euclidean distance between end_effector_pos and the calculated end-effector
        position based on the guess.
    """
    cost = 0.0
    # get the current position
    guess_list = forward_kinematics.fk_foot(guess)[:3,3]

    # find the distance from current position to desired position
    cost = ((guess_list[0]- end_effector_pos[0])**2 + (guess_list[1]- end_effector_pos[1])**2 + (guess_list[2]- end_effector_pos[2])**2) ** (1/2)
    return cost

def calculate_jacobian_FD(joint_angles, delta):
    """
    Calculate the Jacobian matrix using finite differences.

    This function computes the Jacobian matrix for a given set of joint angles using finite differences.

    Args:
        joint_angles (numpy.ndarray), (3,): The current joint angles. A numpy array with 3 elements.
        delta (float): The perturbation value used to approximate the partial derivatives.

    Returns:
        numpy.ndarray: The Jacobian matrix. A 3x3 numpy array representing the linear mapping
        between joint velocity and end-effector linear velocity.
    """
    J = np.zeros((3, 3))


    deltax = [joint_angles[0] + delta, joint_angles[1], joint_angles[2]]
    deltay = [joint_angles[0], joint_angles[1] + delta, joint_angles[2]]
    deltaz = [joint_angles[0], joint_angles[1], joint_angles[2] + delta]

    # the current position
    end_pos = forward_kinematics.fk_foot(joint_angles)[:3,3]
    
    # position by moving the theta 1 a little
    end_pos_delta = forward_kinematics.fk_foot(deltax)[:3,3]
    # the effect this change in theta 1 has on x (its an estimation dfx / dtheta 1)
    x_effect = (end_pos_delta - end_pos) / delta

    # position by moving the theta 2 a little
    end_pos_delta = forward_kinematics.fk_foot(deltay)[:3,3]
    y_effect = (end_pos_delta - end_pos) / delta

    # position by moving the theta 3 a little
    end_pos_delta = forward_kinematics.fk_foot(deltaz)[:3,3]
    z_effect = (end_pos_delta - end_pos) / delta

    # combining to create Jacobian
    J = np.column_stack([x_effect, y_effect, z_effect])
    return J

def calculate_inverse_kinematics(end_effector_pos, guess):
    """
    Calculate the inverse kinematics solution using the Newton-Raphson method.

    This function iteratively refines a guess for joint angles to achieve a desired end-effector position.
    It uses the Newton-Raphson method along with a finite difference Jacobian to find the solution.

    Args:
        end_effector_pos (numpy.ndarray): The desired XYZ coordinates of the end-effector.
            A numpy array with 3 elements.
        guess (numpy.ndarray): The initial guess for joint angles. A numpy array with 3 elements.

    Returns:
        numpy.ndarray: The refined joint angles that achieve the desired end-effector position.
    """
    previous_cost = np.inf
    cost = 0.0

    for iters in range(MAX_ITERATIONS):
        # Calculate the Jacobian of current joint angles
        J = calculate_jacobian_FD(guess, PERTURBATION)

        # Calculate the residual - the amount we need to move by
        cur_pos = forward_kinematics.fk_foot(guess)[:3,3]
        e = end_effector_pos - cur_pos

        # Compute the step to update the joint angles using the Moore-Penrose pseudoinverse using numpy.linalg.pinv
        J_t = np.linalg.pinv(J)
        # update our guess by a little (move our robot a tiny bit in the right direction)
        guess = guess + np.matmul(J_t, e)
        
        cost = ik_cost(end_effector_pos, guess)
        # Calculate the cost based on the updated guess
        if abs(previous_cost - cost) < TOLERANCE:
            break
        previous_cost = cost
    return guess
