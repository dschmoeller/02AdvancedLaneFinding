# Project 2: Advanced Lane Finding



## **Project Goals: **

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.



## General Information: 

- The submission contains two jupyter notebook files. Implementing all necessary steps for building up the pipeline, parameter optimization and visualization of resuts was rather done in **P2TestImages.ipynb**. In order to provide a nice structure (i.e. modularization) when sticking the pipeline together, **P2VideoPipeline.ipynb** containts respective function defintions. The latter one is doesn´t show/visualize intermediate results but is rather meant to create the output videos and building up the pipeline coneniently.   

- Smooth video vs raw video

- The following documentation/description is strucutred based on the desired rubric outcomes

  

## Camera Calibration: 

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The theoretical background behind camera calibration is to learn a camera model which transforms 3D real world points into the 2D camera space. If this transformation model (including distrubances) is known, one can apply the revere transformation in order to undistort the corresponding camera images, i.e. to filter out the distortions. Finding the camera model is done by solving an optimization problem. One aims to find the camera parameters which describes the transformation between known 3D world points and 2D images points best.

The camera matrix and distortion coefficients in this project have been calculated by utilizing both OpenCV´s methods 

```python
cv.findChessboardCorners() 
cv.calibrateCamera(). 
```

After reading the calibration images and grayscaling them, one can apply **cv.findChessboardCorners()** in order to find the points of interest. Due to the well known chessboard structure, one can manually define where these chessboard corners are supposed to be in the real world. Both point structures, the real world points and the points which have been identified in the 2D calibration image serve as input for **cv.calibrateCamera()**. Like described above, the latter method pretty much solves an optimization problem under the hood in order to provide the camera matrix (model) and the distortion coefficients. The picture below shows the effect of undistortion. 

TODO: 01Undistortion.png



## Pipeline (test images): 

#### 1. Provide an example of a distortion-corrected image. 

The first step in the pipeline in to undistort the raw input image, given the camer matrix and distortion coefficients from calibration. The picutre below shows the undistortion effects. The left image corresponds to the raw test input, the right image is the undistorted version. 

TODO: 02UndistortionTestImg.png



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image. Provide an example of a binary image result.

There are both streams applied parallel on the undistroted image, color and gradient based thresholds. The thresholds are tuned based on the test images. In order to apply the color based threshold, the image input is transformed into the HLS color space due to the fact that the S channel has shown to be robust against lighting deviations. Afterwards, a certain color threshold is defined and applied in order to create a binary image. The corresponding code looks like this. 

```python
# Apply color threshold on the S channel in the HLS color space
# --> 1.) Transformation of the undistorted test image to the HLS space and separate the S channel 
hls = cv2.cvtColor(testImgUndistort, cv2.COLOR_RGB2HLS)
S = hls[:,:,2]
# --> 2.) Define the color threshold and apply the threshold to the S-channel.
#         Results, i.e. whether a certain pixel is within (true) or out of (false) the threhold, are stored in a binary map
colorThresh = (100, 255)
colorBinary = np.zeros_like(S)
colorBinary[(S > colorThresh[0]) & (S <= colorThresh[1])] = 1
```

 The resulting binary image is shown below. 

TODO: 03ColorThresholdBinary.png

For the gradient based thresholds, there are four single binary images created, referring to gradient magnitude, gradient in x and y direction and the actual direction in which the gradient points. The actual calculation is done utilizing the Sobel operation. For a more detailed information, I refer to the "***Function Definitions***" part of "**P2TestImage.ipynb**". Based on these four outcomes, there´s some logic applied which combines (i.e. applies a logical AND operation) the x-gradient binary image with the gradient-direction one. The implementation looks like this

```python
# Apply gradient based thresholds and combine them in order to create a unique gradient based output
# --> 1.) Gradient magnitude
magBinary = mag_thresh(testImgUndistort, sobel_kernel=19, mag_thresh=(10, 30))
# --> 2.) Gradient in x and y direction
xGradBinary = abs_sobel_thresh(testImgUndistort, orient='x', sobel_kernel=11, thresh=(30, 100))
yGradBinary = abs_sobel_thresh(testImgUndistort, orient='y', sobel_kernel=19, thresh=(10, 50))
# --> 3.) Gradient direction
dirBinary = dir_threshold(testImgUndistort, sobel_kernel=17, thresh=(0.7, 1.3))
# --> 4.) Combine the above three findings into one single gradient based output
gradientCombBinary = np.zeros_like(magBinary)
gradientCombBinary[(xGradBinary==1) & (dirBinary==1)] = 1
```

  The outcome of the combined gradient based binary image is shown below. 

TODO: 04GradientThresholdBinary.png

In order to combine the color stream with the gradient based one, there has been a logical OR operator applied. The reason for this operation is to catch both findings from the two different "feature spaces". The final binary image contains both, color and gradient based information. It´s shown below. 

TODO: 05ColorGradientCombined.png



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

One crucial part of lane finding is to identify which pixels/points belong to the left and to the right lanes. This task is easier to solve in an transformed perspective, where both lanes appear parallel (like it´s actually the case in the real world). The idea here is to eyeball where the left and right lane would appear in birds eye view. By choosing destination points which cover a rather big area of the image shape, one can "zoom" in a bit, which pretty means nothing by additionaly applying region of interest (ROI). The corresponding implementation is done utilizing the two OpenCV functions

```
cv2.getPerspectiveTransform()
cv2.warpPerspective(). 
```

The former function takes the eyballed source and destination points as input and returns the desired transormation matrix, which can be feed in the latter method to actually warp an input image. A code snippet of the corrsponding implementation looks like this. 

```python
src = np.float32([[leftBottom[1], leftBottom[0]], 
                  [leftTop[1], leftTop[0]], 
                  [rightTop[1], rightTop[0]], 
                  [rightBottom[1], rightBottom[0]]])
dst = np.float32([[leftBottomTarget[1], leftBottomTarget[0]], 
                  [leftTopTarget[1], leftTopTarget[0]], 
                  [rightTopTarget[1], rightTopTarget[0]], 
                  [rightBottomTarget[1], rightBottomTarget[0]]])
# --> 2.) Calculate the perspective transormation matrix
M = cv2.getPerspectiveTransform(src, dst)
# --> 3.) Apply the transformation matrix
warpedImg = cv2.warpPerspective(colorGradCombBinary, M, imageShape)
```

 The warped binary image created in this way is shown below. Note that both lanes are roughly parallel, which is a reasonable sanity check for the correctness of the transformation matrix. 

TODO: 06WarpedImage.png



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In this step, we aim to find pixels which corresponds to lane lines. Further, the goal is to identify which pixels belong to the right and left lane. In a first step, one can construct a histogram which counts the pixels upwards in each single column (i.e. along the y-direction) This is done in the lower part of the image, assuming that the lane is in fact straight there. The row (i.e the x-position) where the sum of all active (i.e. white) pixels has a max can be considered as lane center point. So, searching for two peaks in the histogram leads to the lane center position for both right and left lane. Using the lane center positions as initial setting, one can further apply a sliding window approach to track the lane along the entire picture. This approach searches for a minimum amount of pixels in a sliding window. Once this minimum amount has been identified within a window, one can assume that the lane is supposed to be within this window and an update of the mean x position can be calculated. Otherwise, i.e. the minimum amount of points hasn´t been identfied, the sliding window is shifted. This way, one can keep track of lanes which are following a curve. Applying both histogram search and sliding window can be used in order to find and track lanes within an image. Later, in the video creation step, one can also leverage the fact that the lane position might not diverge that much from frame to frame. So, it´s reasonable to apply sort of an region of interest by searching for potential lane pixels. This can be done by using the polynomial model fit of the previous frame and search around a certain margin. This speeds up computation time as not always the entire picture has to be investigated. Once lane lane pixels have been identified, it´s desirable to construct a model which actually describes the lane. This is done by a fitting a second degree polynomial to the left and right lane pixels/points respectively. Finding lane pixels and fitting a polynomial is implemented by two functins, "***find_lane_pixels ( )***" and "***fit_polynomial( )*** ". Both functions are copied from the corresponding lecture quizzes and their functionality is described above. For further implementation details, refer to the "***Function Definitions***" part of "**P2TestImage.ipynb**". Results/outcomes of the described concepts are shown below.

TODO: 07FindPixelsFitModel.png  

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

todo: write this point out...

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

todo: write this point out...



## Pipeline (video): 

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

todo: write this point out...



## Discussion: 

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

todo: write this point out...







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
