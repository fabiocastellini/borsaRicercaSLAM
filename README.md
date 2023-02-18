# _Research Project entitled "Mappatura e localizzazione simultanea di ambienti indoor con sensori RGBD" (SLAM of indoor environments exploiting a depth camera)_
##### Author: Fabio Castellini (UniVR 2021-2022)

## _Abstract:_
_This project, funded by University of Verona, is aimed at applying a SLAM (simultaneous localization and mapping) algorithm through the use of an RGBD camera (ASUS Xtion PRO LIVE (http://xtionprolive.com/asus-xtion-pro-live)). The overall system was implemented in ROS and programmed with Python3. It is thought for helping blind people while navigating in unknow indoor environments. My work focused on the visual characterization and semantic description of the indoor environment: walls detection and classification, object detection and classification, and collision avoidance. My data stream was then exploited by a partner of this project, to translate it into 3D spatial audio information that will be heard by the blind subject._

## _Project objectives:_ 
- given an RGBD camera, exploit the sensed depth maps to generate a pointcloud of the explored environment using a SLAM algorithm;
- while the 3D cloud of point is generating, try to find the walls assuming that the subject is in a squared/rectangular room with a maximum of 6 walls (front/back/left/right/ceiling/floor);
- locate and classify the general purpose objects inside the environment;
- quantify and spatially localize the free and occupied 3D space, in order to navigate the room without colliding with objects such as chairs, tables, sofa.

## _Mainly utilized tools:_ 
- ASUS Xtion PRO LIVE depth camera;
- ROS (Noetic Ubuntu 20.04);
- Python3;
- SLAM algorithm: RTAB SLAM.

## _Project phases:_
 ___1. Choosing a SLAM algorithm___

The whole project is based on how the sensed pointclouds are stitched together to create a semi-dense 3D reconstruction of the surrounding environment. So, a Simultaneous Localization And Mapping (SLAM) algorithm was used, in particular I started with ORB SLAM but then switched to RTAB (Real-Time Appearance-Based Mapping) SLAM (http://wiki.ros.org/rtabmap_ros) for ease of integration with the ROS environment. Exploiting the RGBD camera, pointclouds are produced and stitched together based on the detected features to create the complete map of the sensed environment while odometry (pose of the camera) is estimated. The created map (with respect to a fixed reference frame that is the initial pose of the camera) can be saved to be later used for navigation purposes (assuming that the environment is static). 


 ___2. Indoor environment's 3D cloud processing___

While the SLAM algorithm is running, the complete pointcloud is published on a ROS topic and it's used to fit planes with a developed iterative RANSAC algorithm. This way, the 6 main wall are extracted and classified with respect to the initial camera pose. Meaning that the left wall corresponds to the wall the initially was on the left with respect to the initial pose of the camera. The same reasoning can be made for the right, front, back walls; also, ceiling and floor are detected but don't depend on the rotation about the vertical axis.

Another important feature is the object detection and classification. General purpose objects such as chair, table, sofa, plant, laptop, mouse...are detected and classified using the pre-trained YoloV3 CNN. Once the 2D bounding box is retrieved, the coorisponding 3D center of mass is computed exploiting the depth map given by the camera. The idea is to assign a signature of the object in the 3D map produced by SLAM and to update the distance and angle with respect to the camera wheneever the latter is moved. This way, imaging that the camera is holded by a blind person, if he moves the distance to the surrounding objects is updated to avoid as much as possible collisions.

Finally, exploiting the OctoMap package (http://wiki.ros.org/octomap) a real-time representation of the occupied and free 3D space is given. The produced information is used to compute if the cilindric volume around the camera (person) is free or not, paying attention to the potential obstacles at floor level.





 ___3. Real-time visualization and storage of the results___
 
 ___4. How to codify the retrieved information to be usable for blind people___

#### _The following video shows a demo performed in a corridor of the University of Verona:_

https://user-images.githubusercontent.com/76775232/219665091-de5a09a2-e82b-4284-9d29-5439d3c4e9ce.mp4

#### _The following video shows a demo performed in a corridor of the University of Verona:_

https://user-images.githubusercontent.com/76775232/219665111-d7fd63f3-5b09-449b-a71f-351c2f6e546a.mp4


#### _Object detection qualitative example:_
![26](https://user-images.githubusercontent.com/76775232/219668593-0eb3c447-dad2-4269-a4b6-2c39176d683f.png)
