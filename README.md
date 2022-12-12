# High-Throughput Phenotyping of Cotton Plants

Phenotype measurements of a cotton plant (such as height and of a number of cotton bolls) are currently carried out by hand. This project's goal is to allow for automatic detection and measurement of those features. To accomplish its goals, the project implements image processing and machine learning techniques. The images are captured by cameras mounted on a robot or a tractor.

Solution: Currently designing and implementing a graph-based SLAM system to scan and detect the cotton bolls using RGB-D mapping, point cloud estimation techniques, and instance segmentation models.

This is currently an ongoing project!

## Data Acquisition

For this phase an ROS based SLAM System was designed which was assembled on a tractor.

Here's an overview of this system:

Two RGB-D Cameras were installed on a tractor so we would be able to scan the plants from top to bottom.
The cameras were being operated by a ROS system running on NVIDIA's Jetson machines. Using the RGB-D images and IMU information from the cameras we scanned 32 rows of cotton plants at different stages of growth during a 6 month period.

Here's a link to a video demonstrating an instance of the scanning sessions being operated on ROS:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=KzjfbDj-uP8
" target="_blank"><img src="http://img.youtube.com/vi/KzjfbDj-uP8/0.jpg" 
alt="Cotton Plant Phenotyping Data Acquisition System" width="640" height="360" border="10" /></a>

Here's some images of the Data Acquistion phase:

<table>
  <tr>
    <td>Scanning during the day with Natural Lighting</td>
     <td>Scanning at night with Artificial Lighting</td>
  </tr>
  <tr>
    <td><img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/day.jpg" width=336 height=448 ></td>
    <td><img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/night.jpg" width=336 height=448 ></td>
  </tr>
 </table>

And here's some instances of PointClouds obtained from a row during one of the scanning sessions.

PointCloud Obtained from the Top Camera (Zoomed In):
<img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/top.png" alt="Top Camera Example" width="730.667" height="384">

PointCloud Obtained from the Bottom Camera:
<img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/Bottom.png" alt="Bottom Camera Example" width="721.3333" height="282.666666667">

The PointClouds being aligned with a Colored ICP algorithm in Python using the Open3D library:
![alt text](https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/combined.png "The Two Cameras Combined using Colored ICP")

Density Analysis of the Cotton Bolls in the Row for Yeild Analysis of the particular seed used for these plants:
![alt text](https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/statistics.png "Density Analysis of the Cotton Bolls in a Row for Yeild Analysis")

# Organ Detection and Segmentation

We are currently developing 2D and 3D detection and segmentation models. 

Here's some results on **3D Open Boll Detection**:

You can see how the algorithm is detection the Open Bolls even when it is surrounded by branches:
![alt text](https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/OpenBollDetection3_Cropped.gif)

And here's some snapshots of the detection in 3D from different angles:
![alt text](https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/1bollDetection2_Cropped.png)
![alt text](https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/bollDetection3_Cropped.png)
![alt text](https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/bollDetection_Cropped.png)
