#!/usr/bin/env python

# ROS libraries
import rospy
from sensor_msgs.msg import PointCloud2, Image
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import OccupancyGrid
import tf
from std_msgs.msg import String, Time, Bool
from rosgraph_msgs.msg import Clock
from geometry_msgs.msg import Point

# Python libraries
import numpy as np
import open3d as o3d
import math
import pandas as pd
import os
import cv2
from cv_bridge import CvBridge
import glob
import uuid
from beepy import beep
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# My libraries
from mylib.lib_cloud_conversion_between_Open3D_and_ROS import *
from mylib.o3d_fitter import *
from mylib.parameters import *
from mylib.transf_utils import *

bridge = CvBridge()

class RtabSlamPlaneFitter:

    def __init__(self):
        super().__init__()

        rospy.init_node('walls_detection_node')

        # ------------------------
        # definition of publishers objects
        # ------------------------
        self.fitted_planes_pcd_pub = rospy.Publisher('/out/fitted_planes_pcd', PointCloud2, queue_size=100)  # pcd with all fitted planes
        self.fitted_walls_pcd_pub = rospy.Publisher('/out/fitted_walls_pcd', PointCloud2, queue_size=100)    # pcd with fitted walls
        self.human_workspace_pcd_pub = rospy.Publisher('/out/human_workspace_pcd', PointCloud2, queue_size=100)  # pcd of the human's workspace
        self.human_workspace_pcd_pub2 = rospy.Publisher('/out/human_workspace_pcd2', PointCloud2, queue_size=100)  # pcd of the human's workspace

        self.human_workspace_bbox_pub = rospy.Publisher('/out/human_workspace_bbox', PointCloud2, queue_size=100)     # bbox of the human's workspace

        self.map2d_img_pub = rospy.Publisher('/out/map2d_img1', Image, queue_size=100)     # rgb image of the 2d occupancy map 
        self.publishing_map2d = False
        self.walls_img_pub = rospy.Publisher('/out/walls_img', Image, queue_size=100)     # rgb image of the fitted walls

        self.nofloor_pcd_pub = rospy.Publisher('/out/no_floor_pcd', PointCloud2, queue_size=100)  # pcd without floor
        self.nowalls_pcd_pub = rospy.Publisher('/out/no_walls_pcd', PointCloud2, queue_size=100)  # pcd without walls

        self.walls_equations_pub = rospy.Publisher('/out/walls_equations', MarkerArray, queue_size=100)  # walls equations written on rviz
        self.json_walls_equations_pub = rospy.Publisher('out/json_walls_equations', String, queue_size=100)
        self.json_human_workspace_pub = rospy.Publisher('out/json_human_workspace', String, queue_size=100)

        self.all_planes_equations_pub = rospy.Publisher('/out/all_planes_equations', MarkerArray, queue_size=100)  # all planes equations written on rviz

        self.camera_position_pub = rospy.Publisher('/out/camera_position', MarkerArray, queue_size=100) # camera pose marker

        self.publish_rviz = True
        # ------------------------

        # ------------------------
        # definition of subscribers objects
        # ------------------------
        # 26 november mod: rtabslam already segments out floor from the rest, so I take advantage of it!
        # cloud_map = full map; cloud_ground = floor; cloud_obstacles = map without floor
        
        #rospy.Subscriber('/rtabmap/cloud_map', PointCloud2, self.rtab_slam_fullcloud_callback)
        rospy.Subscriber('/pcl_filtering/cloud_map', PointCloud2, self.rtab_slam_fullcloud_callback)

        self.o3d_pcd_tot = o3d.geometry.PointCloud()  
        self.ros_pcd_tot = PointCloud2()
        self.pcd_without_floor = o3d.geometry.PointCloud()  
        self.ros_pcd_floor = PointCloud2()
        self.num_new_points = 0 # init    
        self.obstacle_clusters = [] #list of pointclouds representing the obstacles clusters
        self.list_of_ids = [str(uuid.uuid4()) for k in range(1000)]

        #rospy.Subscriber('/rtabmap/cloud_obstacles', PointCloud2, self.rtab_slam_obstacles_callback)
        #rospy.Subscriber('/rtabmap/cloud_ground', PointCloud2, self.rtab_slam_floor_callback)
        rospy.Subscriber('/pcl_filtering/cloud_ground', PointCloud2, self.rtab_slam_floor_callback)

        self.o3d_pcd_floor = o3d.geometry.PointCloud()

        #rospy.Subscriber('/rtabmap/proj_map', OccupancyGrid, self.proj_map_callback)
        rospy.Subscriber('/rtabmap/grid_prob_map', OccupancyGrid, self.proj_map_callback)
        self.tf_sub = tf.TransformListener()

        # ------------------------

        # Assign parameters
        self.ransac_th = 0.2  # for the other planes, before: 0.2
        self.ransac_th_floor = 0.25 # for the floor plane, before: 0.18
        self.ransac_n = 5
        self.ransac_it = 100 # before: 500

        self.filter_by_depth = False
        self.z_threshold = 6.0 #meters (only used if self.filter_by_depth=True)

        self.planar_tollerance = 0.1
        self.min_points_plane = 300 #before filtering: 1500

        #self.voxel_size = 0.05

        # store plane equations' parameters for future uses and to check if a new plane/wall is found or not
        self.all_planes_eq_params = []       # equations' parameters of all the planes found inside the scene
        # self.walls_ordered_eq_params -> 0 = left; 1 = right; 2 = ceiling; 3 = floor; 4 = front; 5 = back
        self.walls_ordered_eq_params = [[],[],[],[],[],[]]   # ORDERED equations' parameters of "walls" found inside the scene
  
        self.fitted_planes_pcd_o3d = o3d.geometry.PointCloud()  # Open3D pcd of all fitted planes
        self.fitted_walls_pcd_o3d = o3d.geometry.PointCloud()   # Open3D pcd of only walls

        self.o3d_pcd_prev = o3d.geometry.PointCloud()  # Open3D previous iteration obstacle cloud
        self.humanws_bbox_pcd = o3d.geometry.PointCloud()

        self.floor_wall_pcd = o3d.geometry.PointCloud()  # Open3D pcd of floor, read from the "cloud_ground" topic
        self.pcd_without_walls = o3d.geometry.PointCloud() # Open3D pcd without fitted walls

        self.prev_camera_pose = [] #new

        self.act_cam_position = []     # store the ACTUAL camera position vector[x,y,z]
        self.act_cam_orientation = []  # store the ACTUAL camera orientation quaternion [x,y,z,w]
        self.prev_cam_orientation = [] # store the PREVIOUS camera orientation quaternion [x,y,z,w]
        
        self.init_cam_T = [] # store the INITIAL camera to map transformation
        self.init_cam_orientation = [] # store the INITIAL camera orientation [x,y,z,w]
       
        self.first_time = True

        print("\n====================================================================")
        print("To visualize the results, use RVIZ and display:\n"\
              "- All fitted planes PointCloud2 on '/out/fitted_planes_pcd' topic\n"\
              "- All fitted planes equations on '/out/all_planes_equations' topic\n"\
              "- Fitted walls PointCloud2 on '/out/fitted_walls_pcd' topic\n"\
              "- Fitted walls equations on '/out/walls_equations' topic\n"\
              "- Camera position and orientation on '/out/camera_position' topic\n"\
              "- Human workspace cube on 'out/human_workspace_pcd' topic\n"\
              "- Human workspace cube bbox on 'out/human_workspace_bbox' topic")
        print("====================================================================\n")
        

        # 2D occupancy map with added walls, objects and human workspace
        self.map2d = None
        self.res_target_cm = 0.5  # target resolution [cm/cell]
        self.proj_map_width = 0
        self.proj_map_height = 0
        self.origin_cm = []
        self.print = True

        self.use_bag = None # False
        while self.use_bag == None:
            rospy.Subscriber('/use_bag', Bool, self.bag_bool_callback)
            print("Waiting for the /use_bag topic!")

        if self.use_bag:
            # Init time clock for offline bags
            self.act_time = 0
            self.prev_time = 0
            self.prev_print_time = 0
            self.act_timestamp = None
            self.print_period = 1

            #self.tf_folder = os.path.join(os.path.dirname(__file__)) + "/recorded_bags/6_laboratorio/tf_saved"
            #self.tf_folder = "/media/fab/Data/Desktop/ICE_temp/borsaSLAM/17-04-23_bag_altair/1_corridoioAltair/tf_saved"
            self.tf_folder = "/media/fab/Data/Desktop/ICE_temp/borsaSLAM/29-05-23_bag_chairs/bagchair2/tf_saved/"

            print(self.tf_folder)
            if not os.path.exists(self.tf_folder):
                raise TypeError("Tf folder path is wrong!")

            filenames = sorted(glob.glob(self.tf_folder + "/*.json"), key=os.path.getmtime)
            self.all_tf_timestamps = np.array([float(fn.split("_")[-1].replace(".json", "")) for fn in filenames])

            rospy.Subscriber('/clock', Clock, self.clock_callback)
        
        #else: # not using the bag's time (Clock)
        timer_period = 0.1 # [seconds]
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.control_loop) 
    # END init()


    # boolean to know if a bag is used or not
    def bag_bool_callback(self, msg):
        self.use_bag = msg.data

    # clock callback
    def clock_callback(self, msg):
        self.act_timestamp = str(msg.clock.secs) + "." + str(msg.clock.nsecs)
        self.act_time = float(str(msg.clock.secs) + "." + str(msg.clock.nsecs))
      

    # control loop, executed each "self.timer_period" seconds
    def control_loop(self, time):          
        print("[control loop] Started")
        t1 = rospy.get_time() # init loop time
        
        # Retrieve map to camera and camera to map transformations
        if self.use_bag:
            self.T_cam_map, self.act_cam_position, self.act_cam_orientation, _ = get_T_cam_to_map_bag(self.tf_folder, self.all_tf_timestamps, self.act_timestamp)
            self.T_map_cam, _, _, _ = get_T_map_to_cam_bag(self.tf_folder, self.all_tf_timestamps, self.act_timestamp)

        else:
            self.T_cam_map, self.act_cam_position, self.act_cam_orientation, _ = get_T_cam_to_map_tf(camera_frame, rviz_frame, self.tf_sub)
            self.T_map_cam, _, _, _ = get_T_cam_to_map_tf(camera_frame, rviz_frame, self.tf_sub)


        # Check that transforms are meaningful
        if np.array_equal(np.array(self.T_cam_map), np.eye(4)) or np.array_equal(np.array(self.T_map_cam), np.eye(4)):
            print("[Warning] Transformations are not initialized!")

        else: # start the actual control loop

            self.num_new_points = abs( self.ros_pcd_tot.width - len(self.o3d_pcd_tot.points))
            print("Number of new points:", self.num_new_points)

            if self.ros_pcd_tot.width > self.min_points_plane and self.num_new_points > 300:

                self.o3d_pcd_tot = convertCloudFromRosToOpen3d(self.ros_pcd_tot)    
                self.o3d_pcd_floor = convertCloudFromRosToOpen3d(self.ros_pcd_floor)                         
                
                self.fit_planes()

                if self.walls_ordered_eq_params[3] != []: # check that floor has been fitted!
                    floor_found = True
                else:
                    floor_found = False

                # If at least a wall that is not the floor was found
                wall_found = False
                for k,el in enumerate(self.walls_ordered_eq_params):
                    if k != 3 and el != []:
                        wall_found = True
                        break

                if wall_found or floor_found:
                    self.pcd_without_walls = self.get_pcd_without_walls()

                    self.classify_planes()
                    
                    # Fix order of the walls
                    self.fix_walls_order()

                self.num_new_points = 0 # init   
                    

            if self.publish_rviz:
                #self.publish_walls_equations()
                #self.publish_all_planes_equations()
                #self.fitted_planes_pcd_pub.publish(convertCloudFromOpen3dToRos(self.fitted_planes_pcd_o3d, rviz_frame))

                self.nofloor_pcd_pub.publish(convertCloudFromOpen3dToRos(self.pcd_without_floor, rviz_frame))
                self.nowalls_pcd_pub.publish(convertCloudFromOpen3dToRos(self.pcd_without_walls, rviz_frame))                
                         
                # Publish the pcd                
                print("Publishing walls's pointcloud (Rviz)...")
                self.fitted_walls_pcd_pub.publish(convertCloudFromOpen3dToRos(self.fitted_walls_pcd_o3d, rviz_frame))
                
                self.publish_camera_position()

            # Publish json dataframe
            self.publish_json_df()

            # compute and publish (if needed) human's workspace around him, 
            # according to "parameters.py" file in the form of a pointcloud and a bounding box
            self.publish_human_workspace()


            # Publish 2D occupancy map
            if len(self.origin_cm) > 0:
                self.publish_proj_map_img()   

            t2 = rospy.get_time()
            print("[control loop] Ended", t2-t1)
    # END control_loop()
    


    def publish_json_df(self):
        print("publish_json_df")
        # Create pandas dataframe from list "self.walls_ordered_eq_params"
        json_data = []
        wall_names = ['left','right', 'ceiling', 'floor', 'front', 'back']
        
        if self.print: 
            print("Detected walls and distances:")        


        for k,eq in enumerate(self.walls_ordered_eq_params):
            if eq != []:
                a,b,c,d,num_points, plane_center, color_id = list(eq)
                center_x, center_y, center_z = plane_center               

                if len(self.act_cam_position) > 0:
                    cam_x,cam_y,cam_z = self.act_cam_position
                    shortest_dist = self.shortest_point_plane_distance(cam_x,cam_y,cam_z, a,b,c,d)
                else:
                    shortest_dist = None  #tf hasn't been received yet

                print(wall_names[k], "wall; distance:", shortest_dist, "m")
                newlist = [wall_names[k], a,b,c,d, shortest_dist, num_points, center_x, center_y, center_z, color_id]
                json_data.append(newlist)
        print("") 

        if len(json_data) > 0:        
            df_json = pd.DataFrame(json_data, columns=["wall_type", "a", "b", "c", "d", "shortest_distance", "num_points", "plane_center_x", "plane_center_y", "plane_center_z", "color_id"])
            #df_json_string = df_json.to_string(index=False)
            #self.json_walls_equations_pub.publish(df_json_string) #publish a string containing the walls info
            self.json_walls_equations_pub.publish(str(df_json.to_json(orient='index')))

    def publish_human_workspace(self):

        t1 = rospy.get_time()
        
        # Compute and publish human workspace cylinder
        if len(self.act_cam_position) > 0:
            x_cam, y_cam, z_cam = self.act_cam_position
            # ws_width, ws_height come from "parameters.py" file and define the around area of interest for the human
            #x_min, x_max = x_cam-ws_width/2, x_cam+ws_width/2
            #y_min, y_max = y_cam-ws_width/2, y_cam+ws_width/2
            z_max = z_cam + ws_height

            if len(self.o3d_pcd_tot.points) > 0:
                z_min = min(np.asarray(self.o3d_pcd_tot.points)[:,2])  # compute the minimum z of the pcd
            else:
                z_min = z_cam

            # Try to define the workspace as a cylinder
            ws_radius = ws_width/2
            points = []
            for i in range(360):
                theta = np.deg2rad(i)
                points.append([x_cam + ws_radius*np.cos(theta), y_cam + ws_radius*np.sin(theta), z_min])
                points.append([x_cam + ws_radius*np.cos(theta), y_cam + ws_radius*np.sin(theta), z_max])
                        
            if len(points) > 0 and self.publish_rviz:
                self.humanws_bbox_pcd = o3d.geometry.PointCloud()
                self.humanws_bbox_pcd.points = o3d.utility.Vector3dVector(np.array(points))
                self.humanws_bbox_pcd.paint_uniform_color([1,0,0]) 
                self.human_workspace_bbox_pub.publish(convertCloudFromOpen3dToRos(self.humanws_bbox_pcd, rviz_frame)) # publish bbox pcd

            
            # Compute and classify points inside the workspace
            nearest_obst_pts = []  # store the closest clusters' points to the camera

            # Check if there are wall points (not ceiling, not floor) inside human ws
            if len(self.fitted_walls_pcd_o3d.points) > 0:                
                walls_pts = np.asarray(self.fitted_walls_pcd_o3d.points)
                walls_cols = np.asarray(self.fitted_walls_pcd_o3d.colors) 

                for wall_id in [0,1,4,5]: # left/right/front/back
                    wall_col = np.expand_dims(set_of_colors[wall_id], axis=0)
                    ind = np.where((walls_cols == wall_col).all(axis=1))[0]
                    
                    if ind.shape[0] > 0:
                        min_dist = np.inf  # init
                        nearest_pnt = None # init

                        for point in walls_pts[ind,:]:
                            p_x, p_y, p_z = point  
                            dist = np.sqrt((p_x-x_cam)**2 + (p_y-y_cam)**2 + (p_z-z_cam)**2)
                            
                            # check if point is inside the human's define workspace boundaries
                            if dist <= ws_radius and p_z <= z_max:
                                if min_dist > dist:
                                    nearest_pnt = point
                                    min_dist = dist  

                        if nearest_pnt is not None: # append nearest obstacle point
                            _,_,_,_,_, plane_center, _ = list(self.walls_ordered_eq_params[wall_id])
                            nearest_obst_pts.append([nearest_pnt, plane_center, wall_id]) # walls id can be 0,1,4,5 !


            # Check if there are obstacle points (no walls, inside human ws)
            if len(self.pcd_without_walls.points) > 0:  
               
                #min_points = int(len(self.pcd_without_walls.points) / 80) # min points to form a cluster
                #if min_points > 20:
                min_points = 20

                # Perform clustering on the pcd_without_walls cloud          
                obst_clusters = o3d_clustering(self.pcd_without_walls, min_points)
             
                if len(self.obstacle_clusters) == 0:
                    self.obstacle_clusters = obst_clusters
                else:
                    for new_cl in obst_clusters:
                        '''
                        is_new_cluster = True
                        for k, prev_cl in enumerate(self.obstacle_clusters):                                             
                            dists = new_cl.compute_point_cloud_distance(prev_cl)                              

                            if np.mean(dists) < 0.05: # threshold of distance; if true means it's the old cluster, so update it
                                is_new_cluster = False
                                self.obstacle_clusters[k] = new_cl
                                break
                        if is_new_cluster:
                            downpcd = new_cl.voxel_down_sample(voxel_size=0.05) # downsample to fasten
                            self.obstacle_clusters.append(downpcd) # append the new detected cluster
                        '''
                        
                        is_new_cluster = True

                        for k, prev_cl in enumerate(self.obstacle_clusters):                                             

                            dists = new_cl.compute_point_cloud_distance(prev_cl)
                            dists = np.asarray(dists)
                            ind = np.where(dists < 0.2)[0]

                            if len(ind) > 0:
                                new_clust_out = new_cl.select_by_index(ind)  
                                self.obstacle_clusters[k] = new_clust_out
                                is_new_cluster = False
                                break

                        if is_new_cluster:
                            self.obstacle_clusters.append(new_clust_out) # append the new detected cluster
                            #downpcd = new_cl.voxel_down_sample(voxel_size=0.05) # downsample to fasten
                            #self.obstacle_clusters.append(downpcd) # append the new detected cluster

                for clust_id, cluster in enumerate(self.obstacle_clusters):                      
                    min_dist = np.inf 
                    nearest_pnt = None
                    for p in np.asarray(cluster.points):
                        p_x, p_y, p_z = p    
                        dist = np.sqrt((p_x-x_cam)**2 + (p_y-y_cam)**2 + (p_z-z_cam)**2)

                        # check if point is inside the human's define workspace boundaries
                        if dist <= ws_radius and p_z <= z_max:          
                            if min_dist > dist:
                                nearest_pnt = p
                                min_dist = dist  

                    if nearest_pnt is not None:
                        pts = np.asarray(cluster.points)
                        center = np.mean(pts, axis=0)
                        nearest_obst_pts.append([nearest_pnt, center, clust_id+6]) # cluster id starts from 6 (0 to 5 are walls!)
           
            # Initialize json data
            strTmapcam = np.array2string(self.T_map_cam, formatter={'float_kind':lambda x: "%.8f" % x}).replace(' ',',').replace('\n',',').replace(',,',',')
            json_data = []

            # Create and color the "nearest points" pointcloud
            ws_pcd = o3d.geometry.PointCloud()
            ws_pcd2 = o3d.geometry.PointCloud()
            # --- Test to see the colored clusters --- #
            # Add walls points
            for point, color in zip(np.asarray(self.fitted_walls_pcd_o3d.points), np.asarray(self.fitted_walls_pcd_o3d.colors)):
                # neglect floor and ceiling by checking the color        
                if not np.array_equal(color, np.array(set_of_colors[2])) and not np.array_equal(color, np.array(set_of_colors[3])): 
                    p_x, p_y, p_z = point  
                    dist = np.sqrt((p_x-x_cam)**2 + (p_y-y_cam)**2 + (p_z-z_cam)**2)

                    # check if point is inside the human's define workspace boundaries
                    if dist <= ws_radius and p_z <= z_max:                          
                        ws_pcd.points.extend(np.expand_dims(point, axis=0))  # add walls
                        ws_pcd.colors.extend(np.expand_dims(color, axis=0))  # add walls
           
            # Add obstacles points
            for clust_id, cluster in enumerate(self.obstacle_clusters):               
                for p in np.asarray(cluster.points):
                    p_x, p_y, p_z = p   
                    dist = np.sqrt((p_x-x_cam)**2 + (p_y-y_cam)**2 + (p_z-z_cam)**2)

                    # check if point is inside the human's define workspace boundaries
                    if dist <= ws_radius and p_z <= z_max:          
                        ws_pcd.points.extend(np.expand_dims(p, axis=0))  # add obstacles
                        ws_pcd.colors.extend(np.expand_dims(np.array(set_of_random_colors[clust_id+6]), axis=0))             
            t2 = rospy.get_time()
            print("[human ws] ", t2-t1)

            wall_names = ["left","right", "ceiling", "floor", "front", "back"]

            for near_p in nearest_obst_pts:
                nearest, center, clust_id = near_p

                center_str = np.array2string(np.array(center), formatter={'float_kind':lambda x: "%.8f" % x}).replace(' ',',').replace('\n',',').replace(',,',',')
                nearest_str = np.array2string(np.array(nearest), formatter={'float_kind':lambda x: "%.8f" % x}).replace(' ',',').replace('\n',',').replace(',,',',')
                
                if clust_id > 5:
                    clust_type = "obstacle"
                    color = set_of_random_colors[clust_id]
                else:
                    clust_type = "wall-" + wall_names[clust_id]
                    color = set_of_colors[clust_id]

                #if clust_id == 6:
                #    beep(sound="coin")

                ws_pcd2.points.extend(np.expand_dims(np.array(nearest), axis=0))
                #ws_pcd2.points.extend(np.expand_dims(np.array(center), axis=0))
                ws_pcd2.colors.extend(np.expand_dims(np.array(color), axis=0)) 

                json_data.append([self.list_of_ids[clust_id], strTmapcam, center_str, nearest_str, clust_type])

            t2 = rospy.get_time()
            print("[human ws] ", t2-t1)

            if len(json_data) > 0:
                df_json = pd.DataFrame(json_data, columns=["unique_id", "T_map_cam", "center_3d", "nearest_3d", "type"])
                self.json_human_workspace_pub.publish(str(df_json.to_json(orient='index')))
                   
            # Publish pointcloud            
            if len(ws_pcd.points) > 0 and self.publish_rviz:
                self.human_workspace_pcd_pub.publish(convertCloudFromOpen3dToRos(ws_pcd, rviz_frame)) # publish pcd
                self.human_workspace_pcd_pub2.publish(convertCloudFromOpen3dToRos(ws_pcd2, rviz_frame)) # publish pcd
        t2 = rospy.get_time()
        print("[human ws] Published", t2-t1)


    def proj_map_callback(self, msg):
        res_cm_map =  msg.info.resolution*100  # [cm/cell]
        if not self.publishing_map2d: # DO NOT UPDATE WHILE PUBLISHING MAP 2D
            self.proj_map_height = round(msg.info.width*res_cm_map/self.res_target_cm)   # [cells]
            self.proj_map_width = round(msg.info.height*res_cm_map/self.res_target_cm) # [cells]              
            self.origin_cm = [msg.info.origin.position.x*100, msg.info.origin.position.y*100]
     

    def publish_proj_map_img(self): 

        t1 = rospy.get_time()
        
        publish_3d_map = True
        if publish_3d_map:
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False) #works for me with False, on some systems needs to be true
            pcd = o3d.geometry.PointCloud()
            pcd.points = self.fitted_walls_pcd_o3d.points
            pcd.colors = self.fitted_walls_pcd_o3d.colors
            pcd.transform([ [1,  0,  0,  100], [0,  0,  1, 100], [0,  -1,  0, 100], [0,  0, 0, 1]])
            vis.add_geometry(pcd)
            vis.get_render_option()
            vis.poll_events()
            vis.update_renderer()
            o3d_screenshot_mat = vis.capture_screen_float_buffer()
            o3d_screenshot_mat = np.asarray(o3d_screenshot_mat) * 255.0        
            mask = (o3d_screenshot_mat[:,:,0] == 255.0) & (o3d_screenshot_mat[:,:,1] == 255.0) & (o3d_screenshot_mat[:,:,2] == 255.0)
            o3d_screenshot_mat[:,:,:][mask] = [194,194,214]
                
            cv2_img = np.uint8(cv2.cvtColor(o3d_screenshot_mat, cv2.COLOR_BGR2RGB)) # Zoom image

            cy, cx = [ i/2 for i in cv2_img.shape[:-1] ]        
            rot_mat = cv2.getRotationMatrix2D((cx,cy), 0, 1.5)
            cv2_img = cv2.warpAffine(cv2_img, rot_mat, cv2_img.shape[1::-1], flags=cv2.INTER_LINEAR)
        
            self.walls_img_pub.publish(bridge.cv2_to_imgmsg(cv2_img))

            vis.destroy_window()
        # --- end publish_3d_map ---
        
        
        self.publishing_map2d = True    

        map2d_img = np.zeros((self.proj_map_height, self.proj_map_width, 3))
        map2d_img[:,:,0] = np.ones((self.proj_map_height, self.proj_map_width))*214
        map2d_img[:,:,1] = np.ones((self.proj_map_height, self.proj_map_width))*194
        map2d_img[:,:,2] = np.ones((self.proj_map_height, self.proj_map_width))*194

        # Apply a grid of 50cm squares (RVIZ visualizes them wrong but when saving it's ok!!!)
        x_coords = np.linspace(0, int(self.proj_map_width-1),  round((self.res_target_cm * self.proj_map_width)/100), endpoint=True,  dtype=int)
        y_coords = np.linspace(0, int(self.proj_map_height-1), round((self.res_target_cm * self.proj_map_height)/100), endpoint=True, dtype=int)       
        for x in x_coords:
            cv2.line(map2d_img, (int(x),0), (int(x), int(self.proj_map_height-1)), color=[230, 230, 230], thickness=2)
        for y in y_coords:    
            cv2.line(map2d_img, (0,int(y)), (int(self.proj_map_width-1), int(y)), color=[230, 230, 230], thickness=2)

        t2 = rospy.get_time()
        print("[map2d] MAP2D", t2-t1)

        # Get camera position
        cam_x, cam_y, cam_z = self.act_cam_position
        cam_row = int((cam_x*100  - self.origin_cm[0])/self.res_target_cm)
        cam_col = int((cam_y*100  - self.origin_cm[1])/self.res_target_cm)

        # Assign specific colors to identify walls
        th = 8 # threshold for coloring the neighborhood
 
        walls_pts = np.asarray(self.fitted_walls_pcd_o3d.points)
        walls_cols = np.asarray(self.fitted_walls_pcd_o3d.colors)

        # Overrite points that belong to a wall (skip ceiling), with the wall's color

        # To fasten, instead of double for:
        #     for p, c in zip(walls_pts, walls_cols):
        for wall_id in [0,1,3,4,5]:
            if walls_pts.shape[0] > 0:
                
                wall_col = np.expand_dims(set_of_colors[wall_id], axis=0)
                ind = np.where((walls_cols == wall_col).all(axis=1))[0]
                
                if ind.shape[0] > 0:
                    points = walls_pts[ind,:]

                    points_px = []
                    for p in points:
                        points_px.append([int((p[0]*100  - self.origin_cm[0])/self.res_target_cm),
                                          int((p[1]*100  - self.origin_cm[1])/self.res_target_cm)])
                    points_px = np.array(points_px)
                     
                    # Show the polygon that represents the plane with low opacity (as a background)
                    hull = ConvexHull(points_px)                
                    xs,ys = points_px[hull.vertices,0], points_px[hull.vertices,1]
                    contours = np.array([ys,xs]).T
                        
                    overlay = map2d_img.copy()       
                    cv2.fillPoly(overlay, pts = [contours], color=tuple(set_of_colors_255_bgr[wall_id]))

                    alpha = 0.25  # Transparency factor.
                    map2d_img = cv2.addWeighted(overlay, alpha, map2d_img, 1 - alpha, 0)

                    # Scatter points inside the workspace
                    for p in points:                  
                        row = int((p[0]*100  - self.origin_cm[0])/self.res_target_cm)
                        col = int((p[1]*100  - self.origin_cm[1])/self.res_target_cm)  

                        if wall_id != 3:
                            dist = np.sqrt((cam_row - row)**2 + (cam_col - col)**2)
                            if dist < person_height*100:   
                                map2d_img[row-th:row+th, col-th:col+th, :] = np.array(set_of_colors_255_bgr[wall_id])-40 # darkened wall color                    
                            else:
                                map2d_img[row-th:row+th, col-th:col+th, :] = set_of_colors_255_bgr[wall_id] # wall color                    
                        else:
                            map2d_img[row-th:row+th, col-th:col+th, :] = set_of_colors_255_bgr[wall_id] # wall color                    

        '''
        # Represent unidentified object/obstacles RED color gradient
        nowalls_points = np.asarray(self.pcd_without_walls.points)

        if len(nowalls_points) > 0:   
          
            a,b,c,d,_,_,_ = self.walls_ordered_eq_params[3] # get floor equation
            floor_eq = np.array([a,b,c,d])

            # Init colors for the gradient
            light_red = np.array([153,153,255])
            dark_red = np.array([0,0,180])
            obstacle_points = []
            max_dist = 1500 # saturation of the distance
            mix = 0.5

            for p in nowalls_points:      
                row = int((p[0]*100  - self.origin_cm[0])/self.res_target_cm)
                col = int((p[1]*100  - self.origin_cm[1])/self.res_target_cm)              
                
                point = np.array([p[0],p[1],p[2],1])

                dot_product = floor_eq.dot(point)
                if dot_product < 0.25:    # in theory should "<0" but there are noisy measures, so impose a threshold                              
                    map2d_img[row-th:row+th, col-th:col+th, :] = set_of_colors_255_bgr[3] # floor color
                elif dot_product < 2: # filter outliers that are too high (probably missed ceiling's points)                    
                    dist = abs(cam_row - row) + abs(cam_col - col)                   
                    
                    if dist > max_dist:
                        dist = max_dist # saturate distance at "max_dist" pixels (using meters is too slow)

                    # Formula for remapping: new_value = ( (old_value - old_min) / (old_max - old_min) ) * (new_max - new_min) + new_min
                    mix = ( (dist + 1) / (max_dist + 1) ) * (1 - 0) + 0 # [0,5] meters to [0,1] abs.
                    red_intensity = mix*light_red + (1-mix)*dark_red
                    
                    obstacle_points.append([row, col, red_intensity])
        
            # Draw the points above the others!
            th = 5
            for el in obstacle_points:
                row, col, red_intensity = el
                map2d_img[row-th:row+th, col-th:col+th, :] = red_intensity.astype(int)
        '''

        if len(self.obstacle_clusters) > 0:
            a,b,c,d,_,_,_ = self.walls_ordered_eq_params[3] # get floor equation
            floor_eq = np.array([a,b,c,d])
                        
            for clust_id, cluster in enumerate(self.obstacle_clusters):
                for p in np.asarray(cluster.points):
                    row = int((p[0]*100  - self.origin_cm[0])/self.res_target_cm)
                    col = int((p[1]*100  - self.origin_cm[1])/self.res_target_cm)              
                    
                    point = np.array([p[0],p[1],p[2],1])
                    dot_product = floor_eq.dot(point)

                    if dot_product < 0.15:    # in theory should "<0" but there are noisy measures, so impose a threshold                              
                        map2d_img[row-th:row+th, col-th:col+th, :] = set_of_colors_255_bgr[3] # floor color

                    elif dot_product < 2: # filter outliers that are too high (probably missed ceiling's points)                    
                        color = np.array(set_of_random_colors[clust_id+6]) * 255
                        color[0], color[2] = color[2], color[0]

                        dist = np.sqrt((cam_row - row)**2 + (cam_col - col)**2)
                        if dist < person_height*100:   
                            map2d_img[row-th:row+th, col-th:col+th, :] = color
                        #else:
                        #    map2d_img[row-th:row+th, col-th:col+th, :] = [0,0,255]

        else: # no wall has been fitted yet, still publish the unclassified pcd
            points = np.asarray(self.o3d_pcd_tot.points)
            if len(points) > 0:
                for p in points:
                    row = int((p[0]*100  - self.origin_cm[0])/self.res_target_cm)
                    col = int((p[1]*100  - self.origin_cm[1])/self.res_target_cm)                                
                    map2d_img[row-th:row+th, col-th:col+th, :] = [180,180,180] # red color (unidentified object/obstacle)
        t2 = rospy.get_time()

        print("[map2d] MAP2D", t2-t1)
        
        # Assign color for the camera direction
        start_point_base = np.array([cam_x, cam_y, cam_z])

        end_point_base = self.T_cam_map.dot( np.array([0,0,0.4,1]).T )[:3]        
        cam_vector = end_point_base - start_point_base        
        cam_vector_len = np.linalg.norm(cam_vector)*100 # still slightly different in len on the map..         

        for k in range(1,int(cam_vector_len)):
            interp_point_x = start_point_base[0] + k*(end_point_base[0] - start_point_base[0])/cam_vector_len
            interp_point_y = start_point_base[1] + k*(end_point_base[1] - start_point_base[1])/cam_vector_len
            row = int((interp_point_x*100  - self.origin_cm[0])/self.res_target_cm)
            col = int((interp_point_y*100  - self.origin_cm[1])/self.res_target_cm)
            th = 5
            map2d_img[row-th:row+th, col-th:col+th, :] = [0,153,255]
        
        # Assign color for the camera position   
        th = 8  
        map2d_img[cam_row-th:cam_row+th, cam_col-th:cam_col+th, :] = [0,0,255]
    
        t2 = rospy.get_time()
        print("[map2d] MAP2D", t2-t1)

        # Assign color for the human workspace
        human_points = np.asarray(self.humanws_bbox_pcd.points)
        if len(human_points) > 0:   
            th = 4
            for p in human_points:
                row = int((p[0]*100  - self.origin_cm[0])/self.res_target_cm)
                col = int((p[1]*100  - self.origin_cm[1])/self.res_target_cm)
                
                if row > 0 and col > 0:
                    map2d_img[row-th:row+th, col-th:col+th, :] = [0,153,255]
               
        self.map2d_img_pub.publish(bridge.cv2_to_imgmsg(np.uint8(map2d_img)))
        t2 = rospy.get_time()
        print("[map2d] Published MAP2D", t2-t1)
        self.publishing_map2d = False


    def rtab_slam_fullcloud_callback(self, ros_pcd):  
        self.ros_pcd_tot = ros_pcd    

    def get_pcd_without_walls(self, th=0.15): #before: th=0.15):
        walls = len(self.fitted_walls_pcd_o3d.points)
        tot = len(self.o3d_pcd_tot.points)
        wwalls = len(self.pcd_without_walls.points)

        if abs( tot - walls - wwalls ) > 50:      
            dists = self.o3d_pcd_tot.compute_point_cloud_distance(self.fitted_walls_pcd_o3d)              
            dists = np.asarray(dists)
            ind = np.where(dists > th)[0]

            pcd = self.o3d_pcd_tot.select_by_index(ind)
            #downpcd = pcd.voxel_down_sample(voxel_size=0.075)
            #return downpcd
            return pcd
        else:
            return self.pcd_without_walls



    def get_pcd_without_floor(self, th=0.2):        
        wf = len(self.pcd_without_floor.points)
        tot = len(self.o3d_pcd_tot.points)
        f = len(self.floor_wall_pcd.points)

        if abs( tot - f - wf ) > 50:   
            dists = self.o3d_pcd_tot.compute_point_cloud_distance(self.floor_wall_pcd)
            dists = np.asarray(dists)
            ind = np.where(dists > th)[0]
            return self.o3d_pcd_tot.select_by_index(ind) # pcd resulting from "TOTAL - FLOOR"          
        else:
            return self.pcd_without_floor
       

    #def rtab_slam_obstacles_callback(self, ros_pcd): 
    #    self.waiting_slam_pcd_topic = False   
    #    self.o3d_pcd_tot = convertCloudFromRosToOpen3d(ros_pcd)     

    def rtab_slam_floor_callback(self, ros_pcd):  
        self.ros_pcd_floor = ros_pcd
     
    # Function to find distance from point to plane
    def shortest_point_plane_distance(self, x1, y1, z1, a, b, c, d):        
        d = abs((a * x1 + b * y1 + c * z1 + d))
        e = (math.sqrt(a * a + b * b + c * c))
        dist = d/e
        # print("Perpendicular distance is", dist)
        return dist

    # Function to compute the farest plane from the actual camera position
    def get_farest_wall(self, already_stored_eq, eq):

        if list(already_stored_eq) == []:
            return eq
        else:
            x1,y1,z1 = self.act_cam_position
            a1,b1,c1,d1,_,_,_ = list(already_stored_eq)
            a2,b2,c2,d2,_,_,_ = list(eq)

            distance1 = self.shortest_point_plane_distance(x1,y1,z1, a1,b1,c1,d1)
            distance2 = self.shortest_point_plane_distance(x1,y1,z1, a2,b2,c2,d2)

            if distance1 > distance2:
                return already_stored_eq
            else:
                return eq

    # Function to check if point belongs to a plane
    def p_belongs_wall(self, p, wall, dist_th=0.1):
        x,y,z = p
        a,b,c,d,_,_,_ = list(wall)

        if abs(a*x+b*y+c*z+d) < dist_th: #p belongs to wall
            return True
        else:
            return False


    # Function to store and update (farest plane has to be kept) the 6 walls in a ordered manner:
    # ordered_walls_eq -> 0 = left; 1 = right; 2 = ceiling; 3 = floor; 4 = front; 5 = back
    def fix_walls_order(self):
        print("[ORDER] Fixing walls' order...")

        ordered_walls_eq = [[],[],[],[],[],[]]    
        cam_x, cam_y, cam_z = self.act_cam_position

        # 6-12 MOD: no need to make assumption: I have the exact floor plane thanks to transformations or in this case, rtabslam segmentation
        #           So, floor plane is used as the "horizontal reference" even though it can be "strongly non horizontal"
        #horizontal_plane = [0,0,1,0,None,None,None] # equation for plane: z = 0
        floor_plane = self.walls_ordered_eq_params[3] # equation for actual floor plane
        ordered_walls_eq[3] = floor_plane.copy() # floor_plane mustn't be initialized!

        for eq in self.all_planes_eq_params:
            a,b,c,d,_,_,_ = list(eq)
            if self.are_planes_parallel(eq, floor_plane): # "eq" is a horizontal plane
                p_x, p_y = cam_x, cam_y
                if abs(c) > 0.001:
                    p_z = -(a*p_x + b*p_y + d)/c
                else:
                    p_z = np.sign(-(a*p_x + b*p_y + d)/c) * 1000
                if p_z > cam_z: # ceiling wall
                    ordered_walls_eq[2] = self.get_farest_wall(self.walls_ordered_eq_params[2], eq)
                    ordered_walls_eq[2][6] = 2 # overrite color_id!

            else: # "eq" is a vertical plane  (orthogonal to the floor)
                # Problem: there are only 2 horizontal planes, but 4 vertical walls!
                # My solution: the first vertical plane found is assumed to be the front one (reasonable in my opinion);
                #              this way if eq // front --> new_front/back else --> left/right
                if self.walls_ordered_eq_params[0]==[] and self.walls_ordered_eq_params[1]==[] and \
                   self.walls_ordered_eq_params[4]==[] and self.walls_ordered_eq_params[5]==[]:
                    ordered_walls_eq[4] = eq.copy()  #consider the fist detected vertical plane as front wall

                else: #already present a front/front that became back wall!                    
                    front_wall = self.walls_ordered_eq_params[4].copy()
                    back_wall = self.walls_ordered_eq_params[5].copy()

                    if front_wall != []:
                        front_back_ref_plane = front_wall.copy()
                    elif front_wall == [] and back_wall != []:
                        front_back_ref_plane = back_wall                    
                    else:  #means that front_wall == [] and back_wall == [] --> should be impossible!
                        left_wall = self.walls_ordered_eq_params[0].copy()
                        right_wall = self.walls_ordered_eq_params[1].copy()

                        if left_wall != []:
                            if abs(c) > 0.001:
                                c2 = -(a+b)/c
                            else:
                                c2 = np.sign(-(a+b)/c) * 1000  
                            front_back_ref_plane = [1, 1, c2, 0,None, None, None]  

                        elif right_wall != []:
                            if abs(c) > 0.001:
                                c2 = -(a+b)/c
                            else:
                                c2 = np.sign(-(a+b)/c) * 1000  
                            front_back_ref_plane = [1, 1, c2, 0,None, None, None] 
                        else:
                            raise TypeError("Shouldn't happen! (check fix_walls_order function)")

                    # check if "eq" is FRONT or BACK wall
                    if self.are_planes_parallel(eq, front_back_ref_plane): 
                        p_y, p_z = cam_y, cam_z
                        if abs(a) > 0.001:
                            p_x = -(b*p_y + c*p_z + d)/a
                        else:
                            p_x = np.sign(-(b*p_y + c*p_z + d)/a) * 1000    

                        if p_x > cam_x: # front wall
                            ordered_walls_eq[4] = self.get_farest_wall(self.walls_ordered_eq_params[4], eq)                            
                        else: # back wall
                            ordered_walls_eq[5] = self.get_farest_wall(self.walls_ordered_eq_params[5], eq)

                    else:  #"eq" is RIGHT or LEFT wall
                        p_x, p_z = cam_x, cam_z
                        if abs(b) > 0.001:
                            p_y = -(a*p_x + c*p_z + d)/b
                        else:
                            p_y = np.sign(-(a*p_x + c*p_z + d)/b) * 1000    

                        if p_y < cam_y: # right wall
                            ordered_walls_eq[1] = self.get_farest_wall(self.walls_ordered_eq_params[1], eq)
                        else: # left wall
                            ordered_walls_eq[0] = self.get_farest_wall(self.walls_ordered_eq_params[0], eq)
                            
        self.walls_ordered_eq_params = ordered_walls_eq  #update the walls' list
        print("END Fixing walls' order...")


    # Method used to publish in topic '/out/walls_equations', a STRING representing the walls equations
    def publish_walls_equations(self):
        wall_names = ["left","right", "ceiling", "floor", "front", "back"]
    
        # Delete old walls equations
        full_msg = MarkerArray()
        for k,eq in enumerate(self.walls_ordered_eq_params):
            msg = Marker()
            msg.header.frame_id = rviz_frame
            msg.ns = ""
            msg.id = k
            msg.text = ""
            msg.type = msg.TEXT_VIEW_FACING
            msg.action = msg.DELETE

            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = -2, -2, k*0.1
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0,0,0,1

            full_msg.markers.append(msg)        
        self.walls_equations_pub.publish(full_msg)  #publish empty message to refresh correctly the equations!

        # Publish new walls equations
        full_msg = MarkerArray()
        for k,eq in enumerate(self.walls_ordered_eq_params):
            if len(list(eq)) > 0:
                msg = Marker()

                a,b,c,d,_,_,_ = list(eq)
              
                if k == 2 or k == 3: #floor or ceiling are "horizontal"
                    plane_type = "H"
                else:
                    plane_type = "V"

                wall_name = wall_names[k]
                msg.text = wall_name + "("+plane_type + ") := " + str(round(a,3)) + "x + " + str(round(b,3)) + "y + " + str(round(c,3)) + "z + " + str(round(d,3)) + " = 0"

                msg.header.frame_id = rviz_frame
                msg.ns = ""
                msg.id = k
                msg.type = msg.TEXT_VIEW_FACING
                msg.action = msg.ADD
                
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = -2, -2, k*0.1
                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0,0,0,1

                msg.scale.x, msg.scale.y, msg.scale.z = 0.1, 0.1, 0.1
                msg.color.r, msg.color.g, msg.color.b, msg.color.a = set_of_colors[k][0], set_of_colors[k][1], set_of_colors[k][2], 1.0

                full_msg.markers.append(msg)
        self.walls_equations_pub.publish(full_msg)
    # END publish_walls_equations()


    # Method used to publish in topic '/out/all_planes_equations', a STRING representing all the detected planes' equations
    def publish_all_planes_equations(self):

        # Delete old walls equations
        full_msg = MarkerArray()
        for k,eq in enumerate(self.all_planes_eq_params):
            msg = Marker()
            msg.header.frame_id = rviz_frame
            msg.ns = ""
            msg.id = k
            msg.text = ""
            msg.type = msg.TEXT_VIEW_FACING
            msg.action = msg.DELETE

            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = -2, -2, 0.1*k
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0,0,0,1

            full_msg.markers.append(msg)        
        self.all_planes_equations_pub.publish(full_msg)  #publish empty message to refresh correctly the equations!

        # Publish new (all) planes equations
        full_msg = MarkerArray()
        for k,eq in enumerate(self.all_planes_eq_params):
            msg = Marker()

            a,b,c,d,_,_,color_id = list(eq)
            
            # CHECK AND PRINT IF PLANE IS HORIZONTAL/VERTICAL, only if floor_plane, that is the reference, is found:
            if self.walls_ordered_eq_params[3] != []:
                floor_plane = self.walls_ordered_eq_params[3]

                if self.are_planes_parallel(eq, floor_plane): # "eq" is a "horizontal plane"
                    plane_type = "H"
                else:
                    plane_type = "V"
                msg.text = plane_type + "("+str(k) + ") := " + str(round(a,3)) + "x + " + str(round(b,3)) + "y + " + str(round(c,3)) + "z + " + str(round(d,3)) + " = 0"
            else:
                msg.text = "("+str(k) + ") := " + str(round(a,3)) + "x + " + str(round(b,3)) + "y + " + str(round(c,3)) + "z + " + str(round(d,3)) + " = 0"

            msg.header.frame_id = rviz_frame
            msg.ns = ""
            msg.id = k
            msg.type = msg.TEXT_VIEW_FACING
            msg.action = msg.ADD
               
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = -2, -2, 0.1*k
            msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0,0,0,0

            msg.scale.x, msg.scale.y, msg.scale.z = 0.1, 0.1, 0.1
            msg.color.r, msg.color.g, msg.color.b, msg.color.a = set_of_colors[color_id][0], set_of_colors[color_id][1], set_of_colors[color_id][2], 1.0

            full_msg.markers.append(msg)    
        self.all_planes_equations_pub.publish(full_msg)  #publish empy message to refresh correctly the equations!
    # END publish_all_planes_equations()


    # Method used to publish in topic '/out/camera_position', a POINT representing the camera position
    def publish_camera_position(self):
        full_msg = MarkerArray()

        for i in range(2):
            msg = Marker()
            msg.header.frame_id = rviz_frame
            msg.ns = ""
            msg.id = i
            msg.action = msg.ADD

            if i == 0:            
                msg.type = msg.SPHERE
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = self.act_cam_position

                msg.color.r, msg.color.g, msg.color.b, msg.color.a  = 0.9, 0.5, 0.6, 1.0
                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = self.act_cam_orientation     
                msg.scale.x, msg.scale.y, msg.scale.z = 0.1, 0.1, 0.1
            else:
                msg.type = msg.LINE_LIST #msg.ARROW 
                msg.color.r, msg.color.g, msg.color.b, msg.color.a  = 0,1.0,0,1.0
                          
                # Very important step: center must be transformed to MAP REFERENCE FRAME!!!!
                direction = [0,0,0.3,1]
                new_direction = self.T_cam_map.dot( np.array(direction) ) 

                new_direction = new_direction[0:3] # remove last coordinate

                p1, p2 = Point(), Point()
                p1.x, p1.y, p1.z = self.act_cam_position
                p2.x, p2.y, p2.z = new_direction

                msg.points.append(p1)
                msg.points.append(p2)

                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = 0,0,0,1 #x,y,z,w  # x_new, y_new, z_new, w_new
                msg.scale.x = 0.01    
            full_msg.markers.append(msg)
        #print("Publishing camera position...")
        self.camera_position_pub.publish(full_msg)
    # END publish_camera_position()




    def fit_planes(self):

        t1 = rospy.get_time()
        
        if self.print:
            print("[FIT PLANES] Starting to fit planes...") 
        
        # Check if fitting the floor plane is needed:        
        num_cloudfloor_pts = len(self.o3d_pcd_floor.points)
        print("# Floor pts", num_cloudfloor_pts)

        fit_floor = False
        # if significant number of points, fit a plane to the floor (basically find the equation...the pcd will be automatically increased by rtabslam)
        if num_cloudfloor_pts > self.min_points_plane/3: 
            if self.walls_ordered_eq_params[3] == []: # never fitted floor
                fit_floor = True

            # if at least "self.min_points_plane" new points
            elif abs(num_cloudfloor_pts - self.walls_ordered_eq_params[3][4]) > 25: #300: 
                fit_floor = True

        if fit_floor:
            print("[FLOOR] Fit floor plane")
            self.floor_wall_pcd, _, floor_model, _, _, floor_center = o3d_plane_fitter(self.o3d_pcd_floor, self.ransac_th_floor, self.ransac_n, self.ransac_it, downsample=False, show_in_outliers=False)
            self.floor_wall_pcd.paint_uniform_color(set_of_colors[3]) #color the pcd relative to the floor

            floor_model.append(len(self.floor_wall_pcd.points)) # add also the number of points of the floor
            floor_model.append(floor_center) # add also the floor_center
            floor_model.append(3) # Fittingadd also the color_id that is 3 because i'm sure it's the floor
            self.walls_ordered_eq_params[3] = floor_model
            
            # Update self.fitted_walls_pcd_o3d
            self.fitted_walls_pcd_o3d.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.fitted_walls_pcd_o3d.points), np.asarray(self.floor_wall_pcd.points)), axis=0))
            self.fitted_walls_pcd_o3d.colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.fitted_walls_pcd_o3d.colors), np.asarray(self.floor_wall_pcd.colors)), axis=0))
            print("[FLOOR] END Fitting floor plane")
        else:
            print("[FLOOR] Not enough points to fit the floor!")


        if self.walls_ordered_eq_params[3] != []: # if already fitted the floor
            # use as temporary copy the pcd without considering the previously fitted floor!
            self.pcd_without_floor = self.get_pcd_without_floor() 
            print("[PCD WITHOUT FLOOR] Num. points:", len(self.pcd_without_floor.points))

            # Try to fit planes until more than "N" points remained (of o3d_pcd_act) 
            self.all_plane_models = [] #initialize
            self.all_plane_clouds = [] #initialize

            o3d_pcd_act = self.pcd_without_floor # temp. copy
            while len(o3d_pcd_act.points) > self.min_points_plane: 
                
                # "Do for the first time" or repeat the fitting process on the "outlier_cloud"
                plane_cloud, outlier_cloud, plane_model, box_cloud, box_points, plane_center = o3d_plane_fitter(o3d_pcd_act, self.ransac_th, self.ransac_n, self.ransac_it, downsample=False, show_in_outliers=False)

                plane_model.append(len(plane_cloud.points)) # add also the number of points that lie on that plane
                plane_model.append(plane_center) # add also the plane_cloud center
                plane_model.append(None) # add also the color_id, at this stage I don't know, so I add None

                self.all_plane_models.append(plane_model)
                self.all_plane_clouds.append(plane_cloud)
            
                o3d_pcd_act = outlier_cloud # consider the new pointcloud to be fitted as the "outlier_cloud"
        
        if self.print:
            t2 = rospy.get_time()
            print("[FIT PLANES] Finished to fit planes...", t2-t1)

    # END fit_planes()

            
    def are_planes_parallel(self, model1, model2):
        # check if "model1" is PARALLEL to "model2"
        a1, b1, c1, _, _, _, _ = model1
        a2, b2, c2, _, _, _, _ = model2        

        #print("parallel coeff=", (a1*a2 + b1*b2 + c1*c2) )
        if abs(a1*a2 + b1*b2 + c1*c2) < 0.5:  #check if the scalar product of the 2 normal vectors is greater than zero  
            return False  #orthogonal planes
        else:
            return True  #parallel planes

    def are_planes_close(self, model1, model2):
        a1, b1, c1, d1, _, center1, _ = model1
        a2, b2, c2, d2, _, center2, _ = model2  

        # if close in terms of model params
        if abs(a1-a2)<self.planar_tollerance and abs(b1-b2)<self.planar_tollerance and abs(c1-c2)<self.planar_tollerance and abs(d1-d2)<self.planar_tollerance:
            return True # CLOSE model (same plane)        
          
        if self.are_planes_parallel(model1, model2):
            # if not close in terms of model params but defines the same wall (due to the presence of forniture)
            
            #point_2_point_dist = math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
            #print("point_2_point_dist=",point_2_point_dist)
            #if point_2_point_dist < 1:  # CLOSE but different (maybe tilted) plane        
                    
            if abs(d1-d2) < 0.3:
                return True  #2 parallel and close planes --> same plane!

        return False

    def is_the_middle_plane(self, potential_middle, model1, model2):
        _, _, _, d1, _, _, _ = model1
        _, _, _, d2, _, _, _ = model2  
        _, _, _, d_potential, _, _, _ = potential_middle

        if (d_potential > d1 and d_potential) < d2 or (d_potential > d2 and d_potential < d1):
            return True
        else:
            return False

    def is_new_wall(self, model_to_check):
        #print("\n\nchecking", len(self.all_planes_eq_params)+1, "plane....\n")
    
        if len(self.all_planes_eq_params) == 0: #check if there are any "wall" models
            #print("[Update] First wall to store!, model=",model_to_check,"color=0") #to debug
            return True, len(self.all_planes_eq_params)

        #elif len(self.all_planes_eq_params) <= 2: 
        else:
            #check if "model" is near a previously found "wall model"
            for k,model in enumerate(self.all_planes_eq_params):               
                if self.are_planes_close(model, model_to_check):
                    #print("Already present wall!, model=",model,"color=",k)  #to debug
                    return False, k
                    
            #print("Storing the wall at", len(self.all_planes_eq_params)) #to debug
            return True, len(self.all_planes_eq_params)

 

    def classify_planes(self): 
        if self.print:
            print("[classify planes] Minimum number of points to form a plane/wall:", self.min_points_plane)

        for p_cloud, p_model in zip(self.all_plane_clouds, self.all_plane_models):

            # a fitted plane should be considered and stored (as wall or secondary plane) if at least "min_points_plane" points lie on it
            if len(p_cloud.points) > self.min_points_plane:
                
                is_new, k = self.is_new_wall(p_model)  #check if it's a new model 
                if is_new: # NEW WALL to be stored
                    p_model[6] = k  #ADD THE WALL ID --> USEFUL FOR COLOR CODING WALLS AND EQUATIONS
                    self.all_planes_eq_params.append(p_model)  #save the wall model
                    
                p_cloud.paint_uniform_color(set_of_colors[k]) #color the plane_cloud 
                
                # Update the pointcloud with colors that refere to the different PLANES (walls+others)
                self.fitted_planes_pcd_o3d.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.fitted_planes_pcd_o3d.points), np.asarray(p_cloud.points)), axis=0))
                self.fitted_planes_pcd_o3d.colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(self.fitted_planes_pcd_o3d.colors), np.asarray(p_cloud.colors)), axis=0))   
             
        num_walls = 0
        for w in self.walls_ordered_eq_params:
            if w != []:
                num_walls += 1
        print("[classify planes] Number of walls/total_planes found:", num_walls, "/", len(self.all_planes_eq_params))  

        
        # Get all the points of the cloud_map pcd and assign colors based on the walls planes they belong
        self.pcd_without_floor = self.get_pcd_without_floor()

        points = np.asarray(self.pcd_without_floor.points)

        temp_total_pcd = self.floor_wall_pcd #o3d.geometry.PointCloud()
    
        for k,wall in enumerate(self.walls_ordered_eq_params):                    
            if k != 3: #skip the floor plane because "pcd_without_floor" doens't include it!
 
                if wall != [] and len(points) > 1:
                    if k == 2:
                        dist_th = 0.1 # higher tolerance for ceiling plane
                    else:
                        dist_th = 0.1

                    temp_pcd = o3d.geometry.PointCloud()
                    belonging_points = []
                    for p in points:
                        if self.p_belongs_wall(p, wall, dist_th):
                            belonging_points.append(p)
                    
                    if len(belonging_points) > 0:
                        temp_pcd.points = o3d.utility.Vector3dVector(np.array(belonging_points))
                        temp_pcd.paint_uniform_color(set_of_colors[k]) #color the temp_pcd 

                        # Update self.fitted_walls_pcd_o3d
                        temp_total_pcd.points = o3d.utility.Vector3dVector(np.concatenate((np.asarray(temp_total_pcd.points), np.asarray(temp_pcd.points)), axis=0))
                        temp_total_pcd.colors = o3d.utility.Vector3dVector(np.concatenate((np.asarray(temp_total_pcd.colors), np.asarray(temp_pcd.colors)), axis=0))
            
        self.fitted_walls_pcd_o3d = temp_total_pcd      
    # END classify_planes()






if __name__ == '__main__':
    
    node = RtabSlamPlaneFitter()
    rospy.spin()







