#!/usr/bin/env python  
import math
import numpy as np
import pandas as pd
import glob
from scipy.spatial.transform import Rotation as R
import rospy
import tf


#---------------------------------------------------------------
# Normalize quaternion
def norm_quat(x, y, z, w):
    norm = np.sqrt(x**2 + y**2 + z**2 + w**2)
    normalized = np.array([x,y,z,w]) / norm
    return normalized.tolist()
#---------------------------------------------------------------



#---------------------------------------------------------------
def quat_2_eul(x=None, y=None, z=None, w=None):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    # To allow single argument quaternion
    if x is not None and y is None and z is None and w is None:
        x,y,z,w = x

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return [roll_x, pitch_y, yaw_z] # in radians
#---------------------------------------------------------------


#---------------------------------------------------------------
def eul_2_quat(roll=None, pitch=None, yaw=None): 
    # To allow single argument quaternion 
    if roll != None and pitch is None and yaw is None:
        roll, pitch, yaw = roll 
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]
#---------------------------------------------------------------


#---------------------------------------------------------------
def rot_2_quat(m):
    #q0 = qw
    t = np.matrix.trace(m)
    q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    if(t > 0):
        t = np.sqrt(t + 1)
        q[3] = 0.5 * t
        t = 0.5/t
        q[0] = (m[2,1] - m[1,2]) * t
        q[1] = (m[0,2] - m[2,0]) * t
        q[2] = (m[1,0] - m[0,1]) * t

    else:
        i = 0
        if (m[1,1] > m[0,0]):
            i = 1
        if (m[2,2] > m[i,i]):
            i = 2
        j = (i+1)%3
        k = (j+1)%3

        t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k,j] - m[j,k]) * t
        q[j] = (m[j,i] + m[i,j]) * t
        q[k] = (m[k,i] + m[i,k]) * t

    return q
#---------------------------------------------------------------


#---------------------------------------------------------------
def quat_2_rot(x=None, y=None, z=None, w=None):  

    if x !=None and  y==None and z==None and w==None:
        x,y,z,w = x

    q0, q1, q2, q3 = x, y, z, w    # Extract the values from Q

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
    return rot_matrix
#---------------------------------------------------------------



def dotproduct(v1, v2):
    return sum((a*b) for a, b in zip(v1, v2))

def vector_length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (vector_length(v1) * vector_length(v2)))





# -------------------------------------------------
# ROS Rerence frames transformations functions (BAG tf or REAL TIME tf)
# -------------------------------------------------
def get_T_cam_to_map_tf(camera_frame, rviz_frame, tf_sub):
    T_camera_map = np.eye(4) 
    trans, rot_eul, rot_quat = [0,0,0], [0,0,0], [0,0,0,0]

    try:
        frame1 = "/"+rviz_frame
        frame2 = "/"+camera_frame

        (trans,rot_quat) = tf_sub.lookupTransform(frame1, frame2, rospy.Time(0))

        x,y,z,w = rot_quat
        rot_eul = quat_2_eul(x,y,z,w)
        
        T_camera_map[:3, :3] = R.from_quat([x,y,z,w]).as_matrix()
        T_camera_map[0:3, 3] = trans
        #print("SUCCESS!!!") # to check
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("[Warning] tf not computed")
    return T_camera_map, trans, rot_quat, rot_eul

def get_T_map_to_cam_tf(camera_frame, rviz_frame, tf_sub):
    T_map_camera = np.eye(4) 
    trans, rot_eul, rot_quat = [0,0,0], [0,0,0], [0,0,0,0]
    try:
        frame1 = "/"+camera_frame
        frame2 = "/"+rviz_frame

        (trans,rot_quat) = tf_sub.lookupTransform(frame1, frame2,  rospy.Time(0))
        x,y,z,w = rot_quat
        rot_eul = quat_2_eul(x,y,z,w)
        
        T_map_camera[:3, :3] = R.from_quat([x,y,z,w]).as_matrix()
        T_map_camera[0:3, 3] = trans
    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        print("[Warning] tf not computed")
    return T_map_camera, trans, rot_quat, rot_eul


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_T_cam_to_map_bag(tf_folder, all_timestamps, timestamp):
    
    # find json tf with the nearest timestamp
    nearest_timestamp = find_nearest(all_timestamps, float(timestamp))    

    base_path = tf_folder + "/" 
    json_path = glob.glob(base_path + "*_" + str(nearest_timestamp) + ".json")[0] # this way more reliable

    try:
        json_df = pd.read_csv(json_path, sep=";") 
    except FileNotFoundError as e:
        print(f""    f"{e}")     
        exit()
    '''
    some_2d_array = np.random.rand(4,4)
    ana = np.array2string(some_2d_array, formatter={'float_kind':lambda x: "%.8f" % x})
    ana = ana.replace(' ',',')
    ana = ana.replace('\n',',')
    ana = ana.replace(',,',',')
    '''
    json_df = json_df.replace(r'\n',',', regex=True)   

    arr_str = json_df["trans_cam_map"][0].replace("[", "").replace("]", "").split(",")
     
    trans_cam_map = np.asarray(arr_str, dtype=float).tolist()

    arr_str = json_df["rot_eul_cam_map"][0].replace("[", "").replace("]", "").split(",")
    rot_eul_cam_map = np.asarray(arr_str, dtype=float).tolist()

    arr_str = json_df["rot_quat_cam_map"][0].replace("[", "").replace("]", "").split(",")
    rot_quat_cam_map = np.asarray(arr_str, dtype=float).tolist()

    arr_str = json_df["T_cam_map"][0].replace("[", "").replace("]", "").split(",")
    T_cam_map = np.asarray(arr_str, dtype=float).reshape((4,4))
  
    return T_cam_map, trans_cam_map, rot_quat_cam_map, rot_eul_cam_map
   

def get_T_map_to_cam_bag(tf_folder, all_timestamps, timestamp):
 
    # find json tf with the nearest timestamp
    nearest_timestamp = find_nearest(all_timestamps, float(timestamp))     
    
    base_path = tf_folder + "/" 
    json_path = glob.glob(base_path + "*_" + str(nearest_timestamp) + ".json")[0] # this way more reliable

    try:
        json_df = pd.read_csv(json_path, sep=";") 
    except FileNotFoundError as e:
        print(f""    f"{e}")     
        exit()
              
    json_df = pd.read_csv(json_path, sep=";") 
    json_df = json_df.replace(r'\n',',', regex=True)     
    
    arr_str = json_df["trans_map_cam"][0].replace("[", "").replace("]", "").split(",")
    trans_map_cam = np.asarray(arr_str, dtype=float).tolist()

    arr_str = json_df["rot_eul_map_cam"][0].replace("[", "").replace("]", "").split(",")
    rot_eul_map_cam = np.asarray(arr_str, dtype=float).tolist()

    arr_str = json_df["rot_quat_map_cam"][0].replace("[", "").replace("]", "").split(",")
    rot_quat_map_cam = np.asarray(arr_str, dtype=float).tolist()

    arr_str = json_df["T_map_cam"][0].replace("[", "").replace("]", "").split(",")
    T_map_cam = np.asarray(arr_str, dtype=float).reshape((4,4))

    
    return T_map_cam, trans_map_cam, rot_quat_map_cam, rot_eul_map_cam
# -------------------------------------------------



