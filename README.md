# High-Throughput Phenotyping of Cotton Plants

Phenotype measurements of a cotton plant (such as height and of a number of cotton bolls) are currently carried out by hand. This project's goal is to allow for automatic detection and measurement of those features. To accomplish its goals, the project implements image processing and machine learning techniques. The images are captured by cameras mounted on a robot or a tractor.

Solution: A graph-based SLAM system to 3D scan, segment, and count the cotton plants and their fruits using RGB-D mapping, point cloud processing techniques, and instance segmentation models. 

Currently working on a publication for this work.

## Data Acquisition

For this phase an ROS based SLAM System was designed which was assembled on a tractor.

Here's a link to a video demonstrating an instance of the scanning sessions being operated on ROS:

<a href="https://www.youtube.com/watch?v=mAAk02cW7og&t=1s
" target="_blank"><img src="https://i.ytimg.com/vi/mAAk02cW7og/maxresdefault.jpg" 
alt="Cotton Plant Phenotyping Data Acquisition System" width="720" height="405" border="10" /></a>

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

# Fruit Segmentation and Enumeration
## 2D
These videos demonstrates our Cotton Fruit Segmentation and Enumeration system. An instance segmentation model was designed and trained to segment the cotton fruits in 4 different stages of growth.
Also, a custom tracking algorithm was implemented to correctly enumerate the number of fruits in each growth-stage throughout the video.

The bounding-boxes are only shown for the objects that are being tracked. These tracked objects are eventually counted (for the right class) if the required conditions are met.

Video captured and processed Mid-Season:
<a href="https://www.youtube.com/watch?v=NsQF2HLmdNE
" target="_blank"><img src="https://i.ytimg.com/vi/NsQF2HLmdNE/maxresdefault.jpg" 
alt="Cotton Plant Phenotyping Data Acquisition System" width="720" height="405" border="10" /></a>

Video captured and processed Late-Season:
<a href="https://www.youtube.com/watch?v=HPllp8o5mxc&t=5s
" target="_blank"><img src="https://i.ytimg.com/vi/HPllp8o5mxc/sddefault.jpg" 
alt="Cotton Plant Phenotyping Data Acquisition System" width="720" height="405" border="10" /></a>

## 3D
### 3D Segmentation by Projection

<a href="https://www.youtube.com/watch?v=xKXTEIwHOng
" target="_blank"><img src="https://i.ytimg.com/vi/xKXTEIwHOng/maxresdefault.jpg" 
alt="Cotton Plant Phenotyping Data Acquisition System" width="720" height="405" border="10" /></a>

### 3D Detection of OpenBolls using Image Processing

You can see how the algorithm is detection the Open Bolls even when it is surrounded by branches:
<img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/OpenBollDetection3_Cropped.gif" alt="" width="722" height="393">

And here's some snapshots of the detection in 3D from different angles:
<img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/1bollDetection2_Cropped.png" alt="" width="720.5" height="393">
<img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/bollDetection3_Cropped.png" alt="" width="721.5" height="392">
<img src="https://github.com/FeriBolour/Cotton_Imaging/blob/main/Images/bollDetection_Cropped.png" alt="" width="721.5" height="393">
