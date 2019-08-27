# Advanced Lane Finding

An image processing pipeline to detect a vehicle's lane on the roadway. The project makes use of two search techiques: "Sliding Windows" and "Around Polynomial Fit Search". This repo is inspired on [Udacity's CarND-Advanced-Lane-Lines repo](https://github.com/udacity/CarND-Advanced-Lane-Lines) abd it is used as a template and guide. 

---
[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./README_images/samples_chess_images.JPG "Chess Images"
[image3]: ./README_images/undistort_output.png "Undistorted".jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

![alt text](README_images/simple_lane_detection.gif)


## Project Steps Overview

- Camera calibration
- Distortion correction
- Color/gradient threshold
- Perspective transform
- Detect lane lines
- Determine the lane curvature and vehicle position with respect to center.
- Output visual display of the lane boundaries and numerical estimation.


## Directory Structure
```
.Simple-Lane-Detection
├── demo.ipynb                   # Main file
├── .gitignore                   # git file to prevent unnecessary files from being uploaded
├── README_images                # Images used by README.md
│   └── ...
├── README.md
└── Udacity_dataset
    ├── examples                 # images used by demo.ipynb and a pair of ground truth videos
    │   └── ...
    ├── test_images              # input images
    │   └── ...
    ├── test_images_output       # output directory generated automatically once you run demo.ipynb
    │   └── ...
    ├── test_videos              # input videos
    │   └── ...
    └── test_videos_output       # output directory generated automatically once you run demo.ipynb
        └── ...
```

## Camera calibration using chessboard images

First step is to take 20 to 30 pictures of a chessboard with the camera you will be using on the self driving car. I have placed these images inside `camera_calibration_images`. Then determine

- nx: Number of corners in any given row
- ny: Number of corners in any given column

<b>corners:</b> Points where two black and two white squares intersect. The following images are samples from `camera_calibration_images`. Note that my chessboard has nx=9 , ny=6. 


With this we define the object points to determine the [camera calibration parameters](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html).

## Distortion Correction



## Apply Perspective Transformation
Next, you want to identify four source points for your perspective transform. The easiest way to do this is to investigate an image where the lane lines are straight. Try experimenting with different src points.


## Demo File

#### Dataset
The demo file makes use of Udacity's dataset to show results. However, once you have run and understood the `demo.ipynb`, feel free to try your own dataset by changing the input directory from the 'Read in Test Image' and 'Read in Test Video' sections from the demo file. The results of your own dataset will be displayed in an utput directory generated automatically by `demo.ipynb` at the same directory level as your input directory.

#### Helper Functions

def draw_roi_box(img, vertices, color=[0, 0, 255], thickness=5):
    """
    Draw a contour around region of interest on img (binary or color)
    Vertices must be 2D array of coordinate pairs [[(x1,y1),...,(x4,y4)]]
    
    """
   

def applyThresh(image, thresh=(0,255)):
    """
    Apply threshold to binary image. Setting to '1' pixels> minThresh & pixels <= maxThresh.
    """
    binary = np.zeros_like(image)
    binary[(image > thresh[0]) & (image <= thresh[1])] = 1
    return binary

def S_channel(image):
    """
    Returns the Saturation channel from an RGB image.
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    return S
    
def sobel_X(image):
    """
    Applies Sobel in the x direction to an RGB image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    abs_sobelx = np.abs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3))
    sobelx     = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    return sobelx

def binary_pipeline(image):
    """
    Combination of color and gradient thresholds for lane detection. 
    Input image must be RGB
    """
def find_lane_pixels_in_sliding_window(binary_warped, nwindows=9, margin=100, minpix=50):
    """
    There is a left and right window sliding up independent from each other.
    This function returns the pixel coordinates contained within the sliding windows
    as well as the sliding windows midpoints
    PARAMETERS
    * nwindows : number of times window slides up
    * margin   : half of window's width  (+/- margin from center of window box)
    * minpix   : minimum number of pixels found to recenter window
    """
    def draw_lane_pixels_in_sliding_window(binary_warped, left_lane_pts, right_lane_pts, window_midpts, margin=100):
    """
    Paints lane pixels and sliding windows.
    PARAMETERS
    * margin : half of window's width  (+/- margin from center of window box)
    """
 def ransac_polyfit(x, y, order=2, n=100, k=10, t=100, d=20, f=0.9):
    """
    RANSAC: finds and returns best model coefficients
    n – minimum number of data points required to fit the model

def fit_polynomial(img_height, left_lane_pts, right_lane_pts):
    """
    Returns pixel coordinates and polynomial coefficients of left and right lane fit.
    If empty lane pts are provided it returns coordinate (0,0) for left and right lane
    and sets fits to None.
    """
  

def find_lane_pixels_around_poly(binary_warped, left_fit, right_fit, margin = 100):
    """
    Returns the pixel coordinates contained within a margin from left and right polynomial fits.
    Left and right fits shoud be from the previous frame.
    PARAMETER
    * margin: width around the polynomial fit
    """
def draw_lane_pixels_around_poly(binary_warped, left_lane_pts, right_lane_pts, previous_fit_pts, margin=100):
    """
    Paints lane pixels and poly fit margins. Poly fit margins are based on previous frame values.
    PARAMETER
    * margin: width around the polynomial fit
    """
 

def augment_previous_fit_pts(left_lane_pts, right_lane_pts, previous_fit_pts, density=4, line_width_margin=10):
    """
    Add to detected points the pts near previous line fits.
    NOTE: This function makes the points from the bottom half of previous line fits five times as dense.
    PARMETERS:
    * density           : number of times points are added near line fits.
    * line_width_margin : range of values generated near line fits
    """  

def measure_curvature_pixels(y_eval, left_fit, right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    PARAMETERS
    * y_eval : where we want radius of curvature to be evaluated (We'll choose the maximum y-value, bottom of image)
    '''
  
  def measure_curvature_meters(y_eval, fit_pts, ym_per_pix= 30/720, xm_per_pix=3.7/900):
    '''
    Calculates the curvature of polynomial functions in meters. 
    NOTE: Chose a straight line birdeye view image to calculate pixel to meter parameters.
    PARAMETERS
    * ym_per_pix : meters per pixel in y dimension (meters/length of lane in pixel)
    * xm_per_pix : meters per pixel in x dimension (meters/width between lanes in pixel)
    * y_eval     : where we want radius of curvature to be evaluated (We'll choose the maximum y-value, bottom of image)
    
  
def lane_center_deviation_meters(img_width,fit_pts, xm_per_pix=3.7/900):
    '''
    Calculates the deviation from the center of the lanes in meters. 
    NOTE: Chose a straight line birdeye view image to calculate pixel to meter parameters.
    PARAMETERS
    * xm_per_pix : meters per pixel in x dimension (meters/width between lanes in pixel)
    '''

def display_lane_roadway(image, fit_pts):
    """
    Colors the roadway of the vehicle's lane defined by the left and right fit points 
    """

def display_corner_image(image, corner_image, scale_size = 1/3):
    """
    Displays an image at the top corner of another image.
    """
    
- `region_of_interest` keeps the region of the image defined by the user while the rest of the image is set to black.
- `ransac_polyfit` finds the best 2nd order model coefficients by randomly testing subsets of a lane's line 
- `line_slope_classifier` classifies pairs of pixel coordinates as left or right lane line based on the pair's line slope. It also allows the user to increase the density of points belonging to left or right lane line.
- `draw_lines` makes use of `region_of_interest`, `ransac_polyfit`, and `line_slope_classifier` to draw the left and right lines of the vehicle's current lane with a user defined color and thickness.
- `draw_roi_box` this is a visual helper. It draws a contour of color around the region of interest to visually identify it. 
- `draw_hough_lines` this is a visual helper. It draws all hough lines with random colors to visually identify them.

#### Pipeline 
This is the process through which each image or video frame goes through.

<p align="center"> <img src="README_images/before.PNG"> </p>
<p align="center"> Fig: Original Image </p>

<p align="center"> <img src="README_images/pipeline.PNG"> </p>
<p align="center"> Fig: Step 1 -> transform to gray.
    Step 2 -> blurr image.
    Step 3 -> apply canny edge detection.
    Step 4 -> obtain region of interest.
    Step 5 -> apply hough transform.
    Step 6 -> draw lines: classify hough lines and apply RANSAC to find polynomial model. </p>

<p align="center"> <img src="README_images/after.PNG"> </p>
<p align="center">Fig: Final image (Note: Step 6 can be applied directly to the color image) </p>


## Drawbacks and improvements

From the challenge video one can identify that the techniques implemented are good for determining lines on on the road but has issues extrapolating those lines. `line_slope_classifier` function needs improvement since it fails when the lane has steep curves. Also the second order polinomial line models for each line should share the same `a` coefficient in `ay^2+by+c` since both lines curve in the same direction. We will deal with this issues in our next repo.





## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
