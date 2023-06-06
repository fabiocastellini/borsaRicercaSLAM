#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import random

def create_marker(m_type, m_id, m_scale, m_pose, points, opacity, rviz_frame):
  marker = Marker()

  marker.header.frame_id = rviz_frame
  marker.header.stamp = rospy.Time.now()  

  # set shape, Arrow: 0; Cube: 1 ; Sphere: 2 ; Cylinder: 3
  marker.type = m_type
  marker.id = m_id
  marker.ns = "basic_shapes"
  marker.action = 0

  # Set the scale of the marker
  marker.scale.x = m_scale[0]
  marker.scale.y = m_scale[1]
  marker.scale.z = m_scale[2]

  # Set the color
  marker.color.r = 0.0
  marker.color.g = 1.0
  marker.color.b = 0.0
  marker.color.a = opacity

  # Set the pose of the marker
  marker.pose.position.x = m_pose[0]
  marker.pose.position.y = m_pose[1]
  marker.pose.position.z = m_pose[2]

  if points is not None:
    for p in points:
      ros_p = Point()
      ros_p.x, ros_p.y, ros_p.z = p
      marker.points.append(ros_p)
  
  marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = 0.0, 0.0, 0.0, 1.0
   
  return marker



def create_bbox(m_id, m_color, box_points, rviz_frame):
  marker = Marker()

  marker.header.frame_id = rviz_frame
  marker.header.stamp = rospy.Time.now()  

  marker.type = 5  # LINE_LIST = 5
  marker.id = m_id
  marker.ns = "basic_shapes"
  
  marker.action = 0 #add/modify marker

  # Set the scale of the marker
  marker.scale.x = 0.01
  marker.scale.y = 1.0
  marker.scale.z = 1.0

  # Set the color
  marker.color.r = m_color[0]
  marker.color.g = m_color[1]
  marker.color.b = m_color[2]
  marker.color.a = 0.5

  marker.pose.orientation.x = 0.0
  marker.pose.orientation.y = 0.0
  marker.pose.orientation.z = 0.0
  marker.pose.orientation.w = 1.0

  # Manually computed point order from open3d bounding box points to line visualization ([0,1] means point0 connects to point1)
  line_order = [[0,1],[0,2],[1,7],[2,7],
                [3,6],[3,5],[6,4],[5,4],
                [0,3],[1,6],[2,5],[7,4]]

  line_points = []
  box_points_arr = np.asarray(box_points)
  for k,order in enumerate(line_order):    
    #print("k", k)
    p1, p2 = Point(), Point()  #create two empty ROS points
    p1.x, p1.y, p1.z = box_points_arr[order[0]][0], box_points_arr[order[0]][1], box_points_arr[order[0]][2]
    p2.x, p2.y, p2.z = box_points_arr[order[1]][0], box_points_arr[order[1]][1], box_points_arr[order[1]][2]
    #print("p1:", p1.x, p1.y, p1.z)
    #print("p2:", p2.x, p2.y, p2.z)

    #append p1 and p2 to the list: point1 should be connected to point2
    line_points.append(p1)  
    line_points.append(p2)

  marker.points = line_points
  
  return marker



def create_planar_bbox(m_id, m_color, box_points, rviz_frame):
    marker = Marker()

    marker.header.frame_id = rviz_frame
    marker.header.stamp = rospy.Time.now()  

    marker.type = 5  # LINE_LIST = 5
    marker.id = m_id
    marker.ns = "basic_shapes"
    
    marker.action = 0 #add/modify marker

    # Set the scale of the marker
    marker.scale.x = 0.01
    marker.scale.y = 1.0
    marker.scale.z = 1.0

    # Set the color
    marker.color.r = m_color[0]
    marker.color.g = m_color[1]
    marker.color.b = m_color[2]
    marker.color.a = 1.0

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    # Manually computed point order from open3d bounding box points to line visualization ([0,1] means point0 connects to point1)
    line_order = [[0,1],[0,3],[3,6],[1,6]]

    line_points = []
    box_points_arr = np.asarray(box_points)
    for k,order in enumerate(line_order):    
      #print("k", k)
      p1, p2 = Point(), Point()  #create two empty ROS points
      p1.x, p1.y, p1.z = box_points_arr[order[0]][0], box_points_arr[order[0]][1], box_points_arr[order[0]][2]
      p2.x, p2.y, p2.z = box_points_arr[order[1]][0], box_points_arr[order[1]][1], box_points_arr[order[1]][2]
      #print("p1:", p1.x, p1.y, p1.z)
      #print("p2:", p2.x, p2.y, p2.z)

      #append p1 and p2 to the list: point1 should be connected to point2
      line_points.append(p1)  
      line_points.append(p2)

    marker.points = line_points
    
    return marker

def get_close_component(corners, th):
    if abs(corners[0,0]-corners[1,0])<th and abs(corners[0,0]-corners[2,0])<th and abs(corners[0,0]-corners[3,0])<th:
        return "x"
    elif abs(corners[0,1]-corners[1,1])<th and abs(corners[0,1]-corners[2,1])<th and abs(corners[0,1]-corners[3,1])<th:
        return "y"
    elif abs(corners[0,2]-corners[1,2])<th and abs(corners[0,2]-corners[2,2])<th and abs(corners[0,2]-corners[3,2])<th:
        return "z"
    return "error"


def create_planar_back_bbox(m_id, m_color, box_points, rviz_frame):
    marker = Marker()
    marker.header.frame_id = rviz_frame
    marker.header.stamp = rospy.Time.now()  
    marker.type = 5  # LINE_LIST = 5
    marker.id = m_id
    marker.ns = "basic_shapes"
    marker.action = 0 #add/modify marker
    marker.scale.x, marker.scale.y, marker.scale.z = 0.01, 1.0, 1.0     # Set the scale of the marker


    # Set the color
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = m_color[0], m_color[1], m_color[2], 1

    marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = 0.0,0.0,0.0,1.0

    line_points = []
    box_points_arr = np.asarray(box_points)
    print("unsortr", box_points_arr)
    
    #box_points_arr = np.sort(box_points_arr, axis = 0)
    prev = Point()
    first = Point()
    for k, p in enumerate(box_points_arr):
        P = Point()
        P.x, P.y, P.z = p

        if k == 0:
          first = P
        else:
            line_points.append(prev)
        line_points.append(P)
        prev = P  
        
    line_points.append(first)

        
    print("pub")


    '''
    distances = []
    for p in box_points_arr:
      distances.append(np.linalg.norm(p))
    distances = np.array(distances)
    min_distances = np.partition(distances,4)[:4]

    closest_corners = []
    for d in min_distances:
      index = np.where(distances == d)
      closest_corners.append( box_points_arr[index] )
    closest_corners = np.array(closest_corners)
    closest_corners = closest_corners[:,0,:]
    
    print("--------------------")
    print(closest_corners)
    print("--------------------")

    # create 4 empty ROS points, my convention:
    #   3 ----- 2
    #   |       |
    #   |       |
    #   0 ----- 1
    p0, p1, p2, p3 = Point(), Point(), Point(), Point()   

    # To understand where the camera is pointed I consider where the constant values are (x/y/z)
    th = 0.001

    depth_direction = get_close_component(closest_corners, th)
    print("DEPTH DIRECTION:", depth_direction)

    if depth_direction == "x":
      max_y = min(closest_corners[0,1], closest_corners[1,1], closest_corners[2,1], closest_corners[3,1])
      min_y = max(closest_corners[0,1], closest_corners[1,1], closest_corners[2,1], closest_corners[3,1])

      max_z = min(closest_corners[0,2], closest_corners[1,2], closest_corners[2,2], closest_corners[3,2])
      min_z = max(closest_corners[0,2], closest_corners[1,2], closest_corners[2,2], closest_corners[3,2])

      for c in closest_corners:
        if abs(c[1] - min_y) < th:    #left
          if abs(c[2] - min_z) < th:      #bottom left
            p0.x, p0.y, p0.z = c
            print("bottom left", c)
          elif abs(c[2] - max_z) < th:    #top left
            p3.x, p3.y, p3.z = c
            print("bott right", c)
        elif abs(c[1] - max_y) < th:  #right
          if abs(c[2] - min_z) < th:      #bottom right
            p1.x, p1.y, p1.z = c
            print("top left", c)
          elif abs(c[2] - max_z) < th:    #top right
            p2.x, p2.y, p2.z = c
            print("top right", c)
    
    elif depth_direction == "y":
      p0.x, p0.y, p0.z = closest_corners[0,:]
      p1.x, p1.y, p1.z = closest_corners[2,:]
      p2.x, p2.y, p2.z = closest_corners[1,:]
      p3.x, p3.y, p3.z = closest_corners[3,:]
    else:
      p0.x, p0.y, p0.z = closest_corners[0,:]
      p1.x, p1.y, p1.z = closest_corners[1,:]
      p2.x, p2.y, p2.z = closest_corners[2,:]
      p3.x, p3.y, p3.z = closest_corners[3,:]

    print(p0.x, p0.y, p0.z)
    print(p1.x, p1.y, p1.z)
    print(p2.x, p2.y, p2.z)
    print(p3.x, p3.y, p3.z)

    line_points.append(p0)
    line_points.append(p1)

    line_points.append(p0)
    line_points.append(p3)

    line_points.append(p1)
    line_points.append(p2)

    line_points.append(p2)
    line_points.append(p3)
    '''
    marker.points = line_points
    
    return marker





def create_bbox_points(m_id, m_color, box_points, obj_center, rviz_frame):
    marker = Marker()

    marker.header.frame_id = rviz_frame
    marker.header.stamp = rospy.Time.now()  

    marker.type = 8  # POINTS = 8
    marker.id = m_id
    marker.ns = "basic_shapes"
    
    marker.action = 0 # add/modify marker

    marker.scale.x, marker.scale.y, marker.scale.z = 0.05, 0.05, 0.05
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = m_color[0], m_color[1], m_color[2], 1
    marker.pose.orientation.x, marker.pose.orientation.y, marker.pose.orientation.z, marker.pose.orientation.w = 0,0,0,1

    # Show box points
    for p in box_points:
        P = Point()
        P.x, P.y, P.z = p
        marker.points.append(P)

    # Show center point
    center = Point()
    center.x, center.y, center.z = obj_center
    marker.points.append(center)
    
    return marker


def create_name_marker(class_name, confidence, distance, m_id, m_color, center, rviz_frame):
    msg = Marker()
    msg.header.frame_id = rviz_frame
    msg.ns = ""
    msg.id = m_id
    msg.text = class_name +" "+ str(int(confidence*100)) + "% " + str(int(distance*100))+"cm"
    msg.type = msg.TEXT_VIEW_FACING
    msg.action = msg.ADD

    # Set the scale of the marker
    msg.scale.x, msg.scale.y, msg.scale.z = 0.15, 0.15, 0.15

    # Set the color
    msg.color.r, msg.color.g, msg.color.b, msg.color.a = m_color[0], m_color[1], m_color[2], 1.0

    msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = center
    msg.pose.position.z += 0.1
    msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0,0,0,1

    return msg
