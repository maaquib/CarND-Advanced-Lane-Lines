## Advanced Lane Finding Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort.png "Undistorted Camera Image"
[image3]: ./output_images/thresholding.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/polyfit.png "Fit Visual"
[image6]: ./output_images/output.png "Output"
[video1]: ./project_video.mp4 "Video"

### Camera Calibration

#### 1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

Image distortion occurs when a camera looks at 3D objects in the real world and transforms them into a 2D image; this transformation isn’t perfect. Distortion actually changes what the shape and size of these 3D objects appear to be. So, the first step in analyzing camera images, is to undo this distortion so that we can get correct and useful information out of them.

I used the OpenCV functions `findChessboardCorners()` to get started. Providing this function with the number of inner corners in the chessboard and a distorted image of a chessboard, it returns an array of detected corners. Using `drawChessboardCorners()` we can draw corners in the sample of chessboard pattern images provided in `camera_cal` folder.

The code for this step is contained in the IPython notebook located at `./AdvancedLaneLineDetection.ipynb#Camera-calibration`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Apply a distortion correction to raw images.

Using the distortion coefficients of the camera calculated as part of the first step, I applied the distortion correction to one of the test images:
![alt text][image2]

#### 2. Use color transforms, gradients, etc., to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at `./AdvancedLaneLineDetection.ipynb#Color/gradient-threshold`).  I experimented with thresholding the yellow color in the saturation channel for better identifying the yellow lanes but did not see a lot of improvement. Here's an example of my output for this step.

![alt text][image3]

#### 3. Apply a perspective transform to rectify binary image ("birds-eye view").

The code for my perspective transform includes a function called `perspective_transform()`, which appears under `./AdvancedLaneLineDetection.ipynb#Perspective-transform` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `perspective_transform()` function takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points are assumed constant.  I chose to hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[ 560., 460.],
     [ 720., 460.],
     [1140., 680.],
     [ 140., 680.]])
offset = 100. # offset for dst points
dst = np.float32(
   [[offset, 0],
    [img_size[0]-offset, 0],
    [img_size[0]-offset, img_size[1]],
    [offset, img_size[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
|  560, 460     |  100, 0       |
|  140, 680     |  100, 720     |
| 1140, 680     | 1180, 720     |
|  720, 460     | 1180, 0       |

I verified that my perspective transform was working as expected by transforming a test image verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Detect lane pixels and fit to find the lane boundary.

The code for detecting the lane lines can be found at `./AdvancedLaneLineDetection.ipynb#Detect-lane-lines`. I used the **Sliding window search** to identify lane lines. The sliding window approach is implemented in `sliding_window()` function. If provided a previous `left_fit` and `right_fight` it will avoid doing a search from scratch, rather use the window obtained from previous lane detection to speed up the lane detection in the new frame. After detecting regions of non-zero pixels in the window, I fit my lane lines with a 2nd order polynomial:

![alt text][image5]

#### 5. Determine the curvature of the lane and vehicle position with respect to center.

The code for radius of curvature can be found in the `Line.set_radius_of_curvature()` function in `./AdvancedLaneLineDetection.ipynb#Define-Line-class`.
```python
def set_radius_of_curvature(self):
    y_eval = np.max(self.ally)
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(self.ally*ym_per_pix, self.allx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    self.radius_of_curvature = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix \
                                      + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```

This function is called every time a new lane is detected as part of the `Line.update_current()` function. The constants for pixels per meter in both the directions are predefined and assumed to be constant (`.AdvancedLaneLineDetection.ipynb#Define-constanst-for-calculating-lane-curvature`). The idea for calculating radius of curvature is simple. The identified pixels are used to fit a 2nd degree polynomial after converting them into meters and using the method defined [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) to calculate the curvature.

#### 6. Warp the detected lane boundaries back onto the original image.

The code for projecting the warped image with identified lanes back to the original image can be found at `./AdvancedLaneLineDetection.ipynb#Project-lane-back-to-the-original-image` under the `project_lane()` function. Here is an example of the result on the test images:

![alt text][image6]

---

### Pipeline (video)

#### 1. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

For making it easier to work with video frames where you want to carry the information from the last frame and use it in the next one, I defined a `Line()` class with utility params and functions. The code can be found after the `get_curvature()` method in `./AdvancedLaneLineDetection.ipynb#Determine-the-lane-curvature`.
The pipeline does the following steps to identify the lane and accounting for failures in a few frames:
1. Un-distort the provided image
2. Generate a binary image with thresholding applied
3. Detect left and right lanes: *Using previous lanes if provided*
4. Update the line objects with current lane detection: *Also calculate radius of curvature and base position*
5. Do a sanity check to find if the detected lanes are valid or not (`sanity_check()` function)
6. If lanes are valid, update the aggregate information by calling the `accept()` method in `Line` class
7. Find deviation from center for the frame
8. Project lane back to original frame with additional metadata

Here's a link to my video result:

[![AdvancedLaneLineDetection](http://img.youtube.com/vi/iy1kDmK0-tw/0.jpg)](http://www.youtube.com/watch?v=iy1kDmK0-tw "AdvancedLaneLineDetection")

---

### Discussion

#### 1. Problems / issues with the implementation of this project
* The thresholding still needs tuning. The pipeline doesn't perform as well on the challenge video. Probably add additional thresholds for yellow and white to identify the lanes better
* Remove noise in the thresholded image by discarding the groups of identified pixels which do not resemble a line
* Further tuning the transform `src` and `dst` points or trying alternative approaches to get the pipeline to work better on the **harder_challenge_vide**. The pipeline performs poorly on this video, swerving around in regions of darkness and lightness. I get the impression that it performs worse on hard turns since the lanes are going out of the region of interest we are transforming
* Refactor the code to better organize the `Line` class. A lot of functions could be better organized in this class
