
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def o3d_plane_fitter(pcd, ransac_th, ransac_n, ransac_it, downsample, show_in_outliers=False):
    """
    Fit plane on a given pointcloud using open3D Python library
   
    Input
        :param pcd:              The input pointcloud (o3d.geometry.PointCloud)
        :param ransac_th:        Ransac algorithm's distance threshold
        :param ransac_n:         Ransac algorithm's number of points randomly sampled to estimate the plane
        :param ransac_it:        Ransac algorithm's number of iterations
        :param downsample:       Boolean parameter to reduce the number of points inside the pointcloud
        :param show_in_outliers: Boolean parameter to plot the sub-cloud to which the plane is fitted using open3D
 
    Output
        :return plane_cloud:   The sub-cloud to which the plane is fitted
        :return outlier_cloud: The remaining pointcloud
        :return plane_model:   The [a,b,c,d] params of the plane equation
        :return obb_cloud:     The oriented bounding box object from the plane_cloud
        :return box_points     The 8 point corners composing the boundin box
    """

    # Downsample the pointcloud, 0.05 is proper for an outdoor scan
    if downsample:
        downpcd = pcd.voxel_down_sample(voxel_size=0.025)
        #print("Downsample the point cloud with a voxel; output #points="+str(len(downpcd.points)))
       
        # Recompute normals of the downsampled point cloud (optional for vivid visualization)
        downpcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    else:
        downpcd = pcd

  
    # Denoise the point cloud
    #cl, denoised_ind = downpcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=5.0)  #to debug removed by now
    #denoised_cloud = downpcd.select_by_index(denoised_ind)                                 #to debug removed by now
 
    #noise_cloud = downpcd.select_by_index(denoised_ind, invert=True) #if we want to visualize noisy/denoised pcds
    #noise_cloud.paint_uniform_color([0, 0, 0])
    #o3d.visualization.draw_geometries([denoised_cloud, noise_cloud])

    # Fit plane
    #pcd = denoised_cloud
    plane_model, inliers = pcd.segment_plane(distance_threshold=ransac_th, ransac_n=ransac_n, num_iterations=ransac_it)
    
    # Extract inliers and outliers pointclouds
    plane_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    #plane_center = pcd.get_center() # before
    #plane_center = plane_cloud.get_center()
    pts = np.asarray(plane_cloud.points)
    plane_center = np.mean(pts, axis=0)

    # Extract plane bounding box as a o3d pointcloud
    obb = plane_cloud.get_oriented_bounding_box()
    
    # Create a pointcloud containing the 8 corners of "obb"
    box_points = obb.get_box_points() 
    obb_cloud = o3d.geometry.PointCloud(box_points)
    
    #print("[o3d_plane_fitter] Extracted number of inlier points = ", len(plane_cloud.points))
    return plane_cloud, outlier_cloud, list(plane_model), obb_cloud, box_points, list(plane_center)


def o3d_clustering(pcd, min_points):
    pcd_num_points = len(pcd.points)
    print("[Clustering] Number of initial points:", pcd_num_points)
    print("[Clustering] Min. number of points to form a cluster", min_points)

    #labels = np.array(pcd.cluster_dbscan(eps=0.001, min_points=min_points, print_progress=False))
    labels = np.array(pcd.cluster_dbscan(eps=0.025, min_points=min_points, print_progress=False))
    max_label = labels.max()
    print(f"[Clustering] Point cloud has {max_label + 1} clusters")

    all_pcds = []
    for i in range(max_label+1):
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[labels == i,:])
        #print("[Clustering] Checking i-th cluster =", i, len(temp_pcd.points))
        all_pcds.append(temp_pcd)

    if len(all_pcds) == 0:
        all_pcds.append(pcd)

    return all_pcds

