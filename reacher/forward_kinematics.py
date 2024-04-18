import math
import numpy as np
import copy

HIP_OFFSET = 0.0335
UPPER_LEG_OFFSET = 0.10 # length of link 1
LOWER_LEG_OFFSET = 0.13 # length of link 2

def rotation_matrix(axis, angle):
  """
  Create a 3x3 rotation matrix which rotates about a specific axis

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians

  Returns:
    3x3 rotation matrix as a numpy array
  """


  rot_mat = np.eye(3)
  rot_mat[0][0] = np.cos(angle) + pow(axis[0], 2) * (1 - np.cos(angle))
  rot_mat[0][1] = axis[0] * axis[1] * (1 - np.cos(angle)) - axis[2] * np.sin(angle)
  rot_mat[0][2] = axis[0] * axis[2] * (1 - np.cos(angle)) + axis[1] * np.sin(angle)

  rot_mat[1][0] = axis[0]*axis[1] * (1-np.cos(angle)) + axis[2]*np.sin(angle)
  rot_mat[1][1] = np.cos(angle) + np.power(axis[1], 2) * (1-np.cos(angle))
  rot_mat[1][2] = axis[1]*axis[2] * (1-np.cos(angle)) - axis[0]*np.sin(angle)
  
  rot_mat[2][0] = axis[2]*axis[0] * (1-np.cos(angle)) - axis[1]*np.sin(angle)
  rot_mat[2][1] = axis[2]*axis[1] * (1-np.cos(angle)) + axis[0]*np.sin(angle)
  rot_mat[2][2] = np.cos(angle) + np.power(axis[2], 2) * (1 - np.cos(angle))
  
  return rot_mat

def homogenous_transformation_matrix(axis, angle, v_A):
  """
  Create a 4x4 transformation matrix which transforms from frame A to frame B

  Args:
    axis:  Array.  Unit vector in the direction of the axis of rotation
    angle: Number. The amount to rotate about the axis in radians
    v_A:   Vector. The vector translation from A to B defined in frame A

  Returns:
    4x4 transformation matrix as a numpy array
  """

  rot = rotation_matrix(axis, angle)
  T = np.column_stack((rot, v_A))
  bottom = np.array([0, 0, 0, 1])
  T = np.row_stack((T, bottom))
  return T

def fk_hip(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the hip
  frame given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the hip frame in the base frame
  """

  axis = np.array([0, 0, 1])
  v_A = np.array([0, 0, 0])
  hip_frame = homogenous_transformation_matrix(axis, joint_angles[0], v_A)
  return hip_frame

def fk_shoulder(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the shoulder
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the shoulder frame in the base frame
  """

  hip_frame = fk_hip(joint_angles)
  axis = np.array([0, 1, 0])
  v_A = np.array([0, -1 * HIP_OFFSET, 0])
  temp_frame = homogenous_transformation_matrix(axis, joint_angles[1], v_A)

  shoulder_frame = np.matmul(hip_frame, temp_frame)
  return shoulder_frame

def fk_elbow(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the elbow
  joint given the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the elbow frame in the base frame
  """
  
  shoulder_frame = fk_shoulder(joint_angles)
  axis = np.array([0, 1, 0])
  v_A = np.array([0, 0, UPPER_LEG_OFFSET])
  temp_frame = homogenous_transformation_matrix(axis, joint_angles[2], v_A)

  elbow_frame = np.matmul(shoulder_frame, temp_frame)
  return elbow_frame

def fk_foot(joint_angles):
  """
  Use forward kinematics equations to calculate the xyz coordinates of the foot given 
  the joint angles of the robot

  Args:
    joint_angles: numpy array of 3 elements stored in the order [hip_angle, shoulder_angle, 
                  elbow_angle]. Angles are in radians
  Returns:
    4x4 matrix representing the pose of the end effector frame in the base frame
  """

  elbow_frame = fk_elbow(joint_angles)
  axis = np.array([0, 0, 0])
  v_A = np.array([0, 0, LOWER_LEG_OFFSET])
  temp_frame = homogenous_transformation_matrix(axis, 0, v_A)

  end_effector_frame = np.matmul(elbow_frame, temp_frame)
  return end_effector_frame
