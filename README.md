# _Research Project entitled "Mappatura e localizzazione simultanea di ambienti indoor con sensori RGBD" (SLAM of indoor environments exploiting a depth camera)_
##### Author: Fabio Castellini (UniVR 2021-2022)

## _Abstract:_
_This project, funded by University of Verona, is aimed at applying a SLAM (simultaneous localization and mapping) algorithm through the use of an RGBD camera (ASUS Xtion PRO LIVE (http://xtionprolive.com/asus-xtion-pro-live)). The overall system was implemented in ROS and programmed with Python3. It is thought for helping blind people while navigating in unknow indoor environments. My work focused on the visual characterization and semantic description of the indoor environment: walls detection and classification, object detection and classification, and collision avoidance. My data stream was then exploited by a partner of this project, to translate it into 3D spatial audio information that will be heard by the blind subject._

## _Project objectives:_ 
- given an RGBD camera, exploit the sensed depth maps to generate a pointcloud of the explored environment using a SLAM algorithm;
- while the 3D cloud of point is generating, try to find the walls assuming that the subject is in a squared/rectangular room with a maximum of 6 walls (front/back/left/right/ceiling/floor);
- locate and classify the general purpose objects inside the environment;
- quantify and spatially localize the free and occupied 3D space, in order to navigate the room without colliding with objects such as chairs, tables, sofa.

## _Main utilized tools:_ 
- ASUS Xtion PRO LIVE depth camera;
- ROS (Noetic Ubuntu 20.04);
- Python3;
- SLAM algorithm: RTAB SLAM.

## _Project phases:_


#### _The following video shows a demo performed in a corridor of the University of Verona:_

https://user-images.githubusercontent.com/76775232/219665091-de5a09a2-e82b-4284-9d29-5439d3c4e9ce.mp4

#### _The following video shows a demo performed in a corridor of the University of Verona:_

https://user-images.githubusercontent.com/76775232/219665111-d7fd63f3-5b09-449b-a71f-351c2f6e546a.mp4


#### _Object detection qualitative example:_
![26](https://user-images.githubusercontent.com/76775232/219668593-0eb3c447-dad2-4269-a4b6-2c39176d683f.png)
