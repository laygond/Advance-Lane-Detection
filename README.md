# Advanced Lane Finding

An image processing pipeline to detect a vehicle's lane on the roadway. The project makes use of two search techiques: "Sliding Windows" and "Around Polynomial Fit Search". This repo is inspired on [Udacity's CarND-Advanced-Lane-Lines repo](https://github.com/udacity/CarND-Advanced-Lane-Lines) and it is used as a template and guide. 

---
[//]: # (Image References)

[image1]: ./README_images/around_fit_correction.png
[image2]: ./README_images/around_fit_fail.png
[image3]: ./README_images/binary_perspectives.png
[image4]: ./README_images/binary_pipeline.png
[image5]: ./README_images/camera_parameters.JPG
[image6]: ./README_images/curvature_eq.JPG
[image7]: ./README_images/curvature_radius_udacity_dataset.png
[image8]: ./README_images/display_around_poly.png
[image9]: ./README_images/display_sliding_window.png
[image10]: ./README_images/image_shift.png
[image11]: ./README_images/perspective_transform.png
[image12]: ./README_images/perspectives.png
[image13]: ./README_images/samples_chess_images.JPG
[image14]: ./README_images/sliding_window.png
[image15]: ./README_images/Undistort.JPG
[image16]: ./README_images/pipeline_diagram.JPG


![alt text](README_images/advanced_lane_detection.gif)


## Directory Structure
```
.Advanced-Lane-Detection
├── demo.ipynb                   # Main file
├── .gitignore                   # git file to prevent unnecessary files from being uploaded
├── README_images                # Images used by README.md
│   └── ...
├── README.md
├── camera_calibration_images    # chessboard images used for camera calibration
│   └── ...
└── Udacity_dataset
    ├── test_images              # input images
    │   └── ...
    ├── test_videos              # input videos
    │   └── ...
    └── test_videos_output       # output directory generated automatically once you run demo.ipynb
        └── ...
```
## Demo File

#### Dataset
The demo file makes use of Udacity's dataset to show results. However, once you have run and understood the `demo.ipynb`, feel free to try your own dataset. Make sure to obtain the camera calibration parameters of the camera used for that dataset to apply distortion correction. For test videos, the results of the processed video will be displayed in an utput directory generated automatically by `demo.ipynb` at the same directory level as your input directory.

#### Helper Functions

- `applyThresh` converts a 2 channel image into binary by setting to '1' pixels in between specified values and remaining pixels to '0'.
- `S_channel` returns the saturation channel from an RGB image.
- `sobel_X` applies Sobel in the x direction to an RGB image.
- `binary_pipeline` uses `sobel_X`, `S_channel`, and `applyThresh` to combine color and gradient thresholds for lane detection.
- `find_lane_pixels_in_sliding_window` one of the methods used for finding left and right lane line pixels.
- `draw_lane_pixels_in_sliding_window` is a visual helper for `find_lane_pixels_in_sliding_window`. it displays the windows and color the pixels withing the windows.
- `fit_polynomial` returns pixel coordinates & polynomial coefficients of fits to left and right lane line pixels. If no lane line points are provided, it returns coordinate (0,0) for left and right fit point coordinates and sets fit coefficients to None. This function makes use of `ransac_polyfit`
- `find_lane_pixels_around_poly` one of the methods used for finding left and right lane line pixels.
- `draw_lane_pixels_around_poly` it is a visual helper for `find_lane_pixels_around_poly`. It paints pixels within the search margin around the polynomial fit used on the previous road frame.
- `augment_previous_fit_pts` is used after `find_lane_pixels_around_poly` to also include the pixels from the previous line fits. This gives preference to weight in the previous tendency of the lane lines before `fit_polynomial` is recalculated.
- `measure_curvature_meters` calculates the radius of curvature of a polynomial functions in meters. The point of evaluation is the bottom of the image, the line points closest to the car.
- `measure_curvature_pixels` does the same as the function above but in pixels. It is not used in the project.
- `lane_center_deviation_meters` Calculates the deviation from of the vehicle from the center of the lane in meters. 
- `display_lane_roadway` this is a visual helper. Colors the roadway of the vehicle's current lane given the left and right line points 
- `display_corner_image` this is a visual helper. It displays a downscaled image at the top corner of another image.
- `ransac_polyfit` finds the best 2nd order model coefficients by randomly testing subsets of a lane's line 
- `draw_roi_box` this is a visual helper. It draws a contour of color around the region of interest to visually identify it. 

#### Project's Steps Overview

- Camera calibration
- Distortion correction
- Perspective transform
- Color/gradient threshold
- Detect lane lines
- Determine the lane curvature and vehicle position with respect to center.
- Output visual display of the lane boundaries and numerical estimation.

<p align="center"> <img src="README_images/before.PNG"> </p>
<p align="center"> Fig: Original Image </p>

##  Project's Steps

#### Camera calibration using chessboard images

First step is to take 20 to 30 pictures of a chessboard with the camera you will be using on the self driving car. I have placed these images inside `camera_calibration_images`. Then determine

- nx: Number of corners in any given row
- ny: Number of corners in any given column

<b>corners:</b> Points where two black and two white squares intersect. The following images are samples from `camera_calibration_images`. Note that my chessboard has nx=9 , ny=6. 

![alt text][image13]

With this we define the  `objpoints` and `imgpoints` to determine the [camera calibration parameters](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html). This is shown in code cell 3 from `demo.ipynb`. Our Camera calibration parameters are:

![alt text][image5]


#### Distortion Correction
With the previous parameters we are ready to correct for distortion as shown in code cell 4 from `demo.ipynb`.  
![alt text][image15]


#### Apply Perspective Transformation
Next, you want to identify four source points for your perspective transform. The easiest way to do this is to investigate on an image where the lane lines are straight. I used `draw_roi_box` helper function to visually choose my source src points.

<p align="center"> <img src=[image11]> </p> 
![alt text][image11]

Then choose you destination points to apply perspective transform. In the following figure:
- The left image has the bottom destination points aligned with the top source points so it narrows down in the bottom.
- The right image has the top destination points aligned with the bottom source points so it widens at the top. (We will mainly used this one for this project)

![alt text][image12]


#### Color Transforms & Gradients ->  Thresholded Binary Image.
I used a combination of color and gradient thresholds to generate a binary image. This is seen in code cell 9 from `demo.ipynb`.
- `applyThresh` converts a 2 channel image into binary by setting to '1' pixels in between specified values and remaining pixels to '0'.
- `S_channel` returns the saturation channel from an RGB image.
- `sobel_X` applies Sobel in the x direction to an RGB image.
Combining all these helper functions into my `binary_pipeline` function we obtain:

![alt text][image4]

And our binary pipeline in our perspective tranformed images would look like:

![alt text][image3]

####  Detect Lane Lines: Sliding Windows
Sliding Windows Method is implemented by `find_lane_pixels_in_sliding_window` and displayed by `draw_lane_pixels_in_sliding_window`. It starts by placing to small windows at the bottom of our binary bird's eye view image. The location of the initial left and right windows for the left and right lane lines is done by applying a vertical histogram to determine the maximum x-axis of activated pixels. Next, those windows are slided up and recentered to where most pixels are. This process is repeated until you reach the top of the image. Once the pixels are found you can use `fit_polynomial` to obtain our new fits to each lane line.

![alt text][image14]

####  Detect Lane Lines: Around Polynomial Fit
Once the lane line fits are found we do not have to search from scratch using the sliding windows method again. We can search in the next frame around these previously found line fits. I will first shift our binary image to simulate our "next" frame

![alt text][image10]

Then apply `find_lane_pixels_around_poly`, display it using `draw_lane_pixels_around_poly`, and finally use `fit_polynomial` to obtain our new fits to each lane line. 

![alt text][image2]

In the right lane, the search of pixels around previous fit was a success; however, the new line fit in yellow for the pixels highlighted in blue failed to continue the shape of the previous line fit in purple.

Even though we could apply RANSAC and correct for this, it might be computationally expensive to overtune its parameters. Therefore, we will weight more the previous line fit when finding the new line fit. This can be resolved by artificially adding points around the previous fit before finding the new fit by using `augment_previous_fit_pts`

Let's try again! this time we apply `find_lane_pixels_around_poly`, display it using `draw_lane_pixels_around_poly`, add points using `augment_previous_fit_pts`, annd finally use `fit_polynomial` to obtain our new fits to each lane line. 

![alt text][image1]

It worked!


#### Radius of Curvature &  Deviation of Vehicle from Center of Lane.

Radius of curvature is implemented to each lane line using `measure_curvature_meters` through the following equations:

![alt text][image6]

This is done in code cell 17 from `demo.ipynb`. The results we have acquired for one frame are Left Radius: 1271.87 m  and Right Radius: 240.72 m. These values are constantly changing from frame to frame since the road curvature changes. However, we can observe that they oscillate around 1 Km and it can be corroborated in the following google maps image:

![alt text][image7]

Finally, for finding the deviation of the vehicle from the center of the lane we assume the camera of the car is positioned at its center. Therefore, the difference between the center of the image and the center of the lane is the deviation we are looking for. This is done by `lane_center_deviation_meters` in code cell 18 from `demo.ipynb`. The following subsection displays some results.

#### Output Visual Display

This subsection is in charge of adding all the results from the previous subsections. All that binary birds eye view analysis will be displayed on the top corner of the original image using `display_corner_image`. With the left and right fit points we can color to the vehicle's lane and warp it back to the original image which is done by `display_lane_roadway`. Finally, radius of curvature, lane deviation, and search type algorithm implemented can be displayed using `cv2.putText`.

![alt text][image9] ![alt text][image8]

We put the same image twice through our pipeline. In the left it used sliding window search while in the right since it already had the previous line fit, it used around poly fit search.

## Pipeline Strategy 

![alt text][image16]

Here's a [link to my video result](./Udacity_dataset/test_videos_output/project_video.mp4)

## Drawbacks and improvements

There is a lot of room for impovement as it can be seen in the challenge videos. The current image processing pipeline fails due to weak sanity checks and poor binary pipeline processing. Also both perspective transform should be used to make it more robust. We will deal with this issues in our next repo.

Possible indicators for sanity check:
- Area in between lane changes gradually even if your lane gets narrow or wider
- Minimum width of lane for car should be respected
- Perhaps use the focus and directrix of parabolas as extra indicators

General Improvements:
- Fit both parabolas at the same time rather than independantly; in this way the leading coefficient `a` in `ay^2+by+c`of the parabolas are shared.
- Create window-grids inside the polyfit search area to control the amount of activated pixels according to what a line's pixel density should have. 
- Use Neural Nets to make a cleaner binary image