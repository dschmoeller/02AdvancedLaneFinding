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

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/01Undistortion.png)

![](output_images\01Undistortion.png)



## Pipeline (test images): 

#### 1. Provide an example of a distortion-corrected image. 

The first step in the pipeline in to undistort the raw input image, given the camer matrix and distortion coefficients from calibration. The picutre below shows the undistortion effects. The left image corresponds to the raw test input, the right image is the undistorted version. 

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/02UndistortionTestImg.png)

![](output_images\02UndistortionTestImg.png)



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

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/03ColorThresholdBinary.png)

![](output_images\03ColorThresholdBinary.png)

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

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/04GradientThresholdBinary.png)

![](output_images\04GradientThresholdBinary.png)

In order to combine the color stream with the gradient based one, there has been a logical OR operator applied. The reason for this operation is to catch both findings from the two different "feature spaces". The final binary image contains both, color and gradient based information. It´s shown below. 

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/05ColorGradientCombined.png)

![](output_images\05ColorGradientCombined.png)



#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

One crucial part of lane finding is to identify which pixels/points belong to the left and to the right lanes. This task is easier to solve in an transformed perspective, where both lanes appear parallel (like it´s actually the case in the real world). The idea here is to eyeball where the left and right lane would appear in birds eye view. By choosing destination points which cover a rather big area of the image shape, one can "zoom" in a bit, which pretty means nothing by additionaly applying region of interest (ROI). The corresponding implementation is done utilizing the two OpenCV functions

```
cv2.getPerspectiveTransform()
cv2.warpPerspective(). 
```

The former function takes the eyballed source and destination points as input and returns the desired transormation matrix, which can be feed in the latter method to actually warp an input image. A code snippet of the corrsponding implementation looks like this. 

```python
![06WarpedImage](C:\Users\Schmoeller\Desktop\Bodi\Job\01 ARGO AI\Udacity Self Driving Car Engineer\02FindingLaneLinesAdvanced\02AdvancedLaneFinding\CarND-Advanced-Lane-Lines\output_images\06WarpedImage.png)![06WarpedImage](C:\Users\Schmoeller\Desktop\Bodi\Job\01 ARGO AI\Udacity Self Driving Car Engineer\02FindingLaneLinesAdvanced\02AdvancedLaneFinding\CarND-Advanced-Lane-Lines\output_images\06WarpedImage.png)src = np.float32([[leftBottom[1], leftBottom[0]], 
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

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/06WarpedImage.png)

![](output_images\06WarpedImage.png)



#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In this step, we aim to find pixels which corresponds to lane lines. Further, the goal is to identify which pixels belong to the right and left lane. In a first step, one can construct a histogram which counts the pixels upwards in each single column (i.e. along the y-direction) This is done in the lower part of the image, assuming that the lane is in fact straight there. The row (i.e the x-position) where the sum of all active (i.e. white) pixels has a max can be considered as lane center point. So, searching for two peaks in the histogram leads to the lane center position for both right and left lane. Using the lane center positions as initial setting, one can further apply a sliding window approach to track the lane along the entire picture. This approach searches for a minimum amount of pixels in a sliding window. Once this minimum amount has been identified within a window, one can assume that the lane is supposed to be within this window and an update of the mean x position can be calculated. Otherwise, i.e. the minimum amount of points hasn´t been identfied, the sliding window is shifted. This way, one can keep track of lanes which are following a curve. Applying both histogram search and sliding window can be used in order to find and track lanes within an image. Later, in the video creation step, one can also leverage the fact that the lane position might not diverge that much from frame to frame. So, it´s reasonable to apply sort of an region of interest by searching for potential lane pixels. This can be done by using the polynomial model fit of the previous frame and search around a certain margin. This speeds up computation time as not always the entire picture has to be investigated. Once lane lane pixels have been identified, it´s desirable to construct a model which actually describes the lane. This is done by a fitting a second degree polynomial to the left and right lane pixels/points respectively. Finding lane pixels and fitting a polynomial is implemented by two functins, "***find_lane_pixels ( )***" and "***fit_polynomial( )*** ". Both functions are copied from the corresponding lecture quizzes and their functionality is described above. For further implementation details, refer to the "***Function Definitions***" part of "**P2TestImage.ipynb**". Results/outcomes of the described concepts are shown below.

  ![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/07FindPixelsFitModel.png)

  ![](output_images\07FindPixelsFitModel.png)



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The equation for the radius of curvature is very well known and depends on the parameters of the polynomial model which has been fitted to the lanes previously. So feeding the parameters into the formula leads to the desired result. It´s worth to mention that curvature information needs to be expressed in meters in order to reflect the real world circumstances. That´s why there´s a conversion constant applied in the equation to transform from pixel space into a metric space. Besides the model parameters, there has to be a y-value provided. This value corresponds to the point of interest. Given the fact than one is typically interested in the curvature which the car currently drives, a reasonable choice is to use the maximum y value. This means, the curvature is calculated at the bottom of the picture, i.e. where the actual car would have been located. The corresponding implmenetation looks like this. Note that it does make sense to provide a unique value for the curvature. This could either be done by averaging both values for left and right lane or choose the more certain one (e.g. if the left lane is solid, there are potentially more lane pixels identified). 

 

```
def measure_curvature_real(ploty, left_fit_cr, right_fit_cr):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad
```

In real life, the car is not always perfectly driving exactly in the middle of the lane. That´s why the information how much the car is shifted might be valuable. Assuming that the camera is mounted in the middle of the car, one can further assume that the actual lane center point is located at the middle of the picture (in x direction). This means, if the car would drive perfectly in the middle of the lane, both lane lines are equally distanced from the image center point. This assumption leads to the following implementation. Note that the same coefficient applies as above in order to express the lane center offset in meters rather than in pixel space.   

```
# m/pixel 
xm_per_pix = 3.7/700 # meters per pixel in x dimension

# The x-middle of the image (in meter)
xImgCenterPoint = imageShape[0]/2 * xm_per_pix

# The actual lane center points (Acquired from the polynomial model)
leftLaneCenterPoint = left_fitx[-1]*xm_per_pix
rightLaneCenterPoint = right_fitx[-1]*xm_per_pix
centerPoint = (leftLaneCenterPoint + rightLaneCenterPoint)/2

# The deviation between lane center point and image center point measures how much the car shifted away from the lane center
# Right offset should be considered as positve shift
laneOffset = centerPoint - xImgCenterPoint
```



#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

Classification, lane matching and curvature calculation has been done in the birds eye view space. However, the left and right lane position i.e. the respective lane models have to be projected back into the actual camera space. Therefore one can calculate and apply the inverse transformation matrix from above. The final outcome containing lane area, curvatures and lane offset information projected onto the initial input image is shown below. 

![](https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_images/08LaneAreaProjected.png)

![](output_images\08LaneAreaProjected.png)



## Pipeline (video): 

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

For the video pipeline implementation in "**P2VideoPipeline.ipynb**", there have been some features added to the basic version from "**P2TestImage.ipynb**". One major distinction is the partial modularization of the code. This enables a very convenient (and user friendly) way to build up the basic pipeline, as shown in the code snippet  below. 

```python
def process_image(image):
    # 1 Apply color and gradient based thresholds
    colorGradientFilteredImg = applyColorGradientThresholds(image)
    # 2 Warp image into birds eye view
    warpedImg, M = applyBirdsEyeTransform(colorGradientFilteredImg)
    # 3 Extract the lane points and fit a polyonomial model 
    ploty, left_fitx, right_fitx, left_fit_coeffs, right_fit_coeffs, leftPts, rightPts = fitLaneModel(warpedImg)
    # 4 Calculate curvature and lane offset
    left_curv_real, right_curv_real = measure_curvature_real(ploty, left_fit_coeffs, right_fit_coeffs)
    curvature = (left_curv_real + right_curv_real)/2
    laneOffset = calculate_lane_offset(left_fitx[-1], right_fitx[-1])
    # 5 Render lane area and back transformation to camera view space 
    lanesInImg = transformBackToCameraView(warpedImg, image, M, left_fitx, right_fitx, ploty, leftPts, rightPts)
    # 6 Add curvature and lane center information
    renderCurvatureAndLaneOffset(lanesInImg, curvature, laneOffset)
    return lanesInImg
```

The outcome of this basic approach looks promising but isn´t very robust at locations where shadowing effects occur or the line markings are not clearly visible. Also, the above approach doesn´t take advantage of the polynomial region of interest for searching lane points, like discussed above. Adding the polynomial based search was rather straight forward, since the code from the lession quiz could be utilizied with small adaptions. The sanity check is supposed to filter out flickering lane models. The idea is to check whether a certain lane finding (i.e. lane model) seems to be reasonable. Whenver a lane model was identified as invalid, this information gets dropped. This means that the information from the previous valid frame is used instead. There are two conditions which are used in order to prove validity. The first one leverages the fact that one has a prior belief of how the polynomial model looks like, i.e. what reasonable values for the corresponding paramters (a, b, c) are. So, one can directly check whether a certain lane model parameter is greater than a defined threshold. The second check builds upon the assumption that the lane model parameters don´t rapidly change from the previous frame to the current one. So, one can check the deviation against a certain threshold in order to identify outliers. The defintion of this extended pipeline is shown above. Note that the suggested lane class from the lecture is used to keep track of the current lane findings, which is iteratively updated in the "**sanityCheck( )**" function.  

```python
def process_image_smooth(image):
    # 1 Apply color and gradient based thresholds
    colorGradientFilteredImg = applyColorGradientThresholds(image)
    # 2 Warp image into birds eye view
    warpedImg, M = applyBirdsEyeTransform(colorGradientFilteredImg)
    # 3 Extract the lane points and fit a polyonomial model
    if leftLane.detected == False and rightLane.detected == False: 
        slidingWindow = True
    else:
        slidingWindow = False
    left_fit_prev = leftLane.current_fit
    right_fit_prev = rightLane.current_fit
    ploty, left_fitx, right_fitx, left_fit_coeffs, right_fit_coeffs, leftPts, rightPts = fitLaneModel(warpedImg, left_fit_prev, right_fit_prev, slidingWindow)
    # 4 Calculate curvature and lane offset
    left_curv_real, right_curv_real = measure_curvature_real(ploty, left_fit_coeffs, right_fit_coeffs)
    curvature = (left_curv_real + right_curv_real)/2
    laneOffset = calculate_lane_offset(left_fitx[-1], right_fitx[-1])
    # 5 Run sanity check over findings in this particular image and update the lane class information accordingly
    leftLaneFindings = [left_fitx, left_curv_real, left_fit_coeffs, leftPts]
    rightLaneFindings = [right_fitx, right_curv_real, right_fit_coeffs, rightPts]
    sanityCheck(leftLaneFindings, rightLaneFindings)
    # 6 Render lane area and back transformation to camera view space 
    # Take lane information which is tracked in the lane classes
    left_fitx = leftLane.recent_xfitted 
    right_fitx = rightLane.recent_xfitted
    leftPts = leftLane.allx
    rightPts = rightLane.allx
    curvature = (leftLane.radius_of_curvature + rightLane.radius_of_curvature)/2
    laneOffset = calculate_lane_offset(left_fitx[-1], right_fitx[-1])
    lanesInImg = transformBackToCameraView(warpedImg, image, M, left_fitx, right_fitx, ploty, leftPts, rightPts)
    # 7 Add curvature and lane center information
    renderCurvatureAndLaneOffset(lanesInImg, curvature, laneOffset)
    return lanesInImg
```

A link to the final video outcome is provided here: 

[]: https://github.com/dschmoeller/02AdvancedLaneFinding/blob/master/CarND-Advanced-Lane-Lines/output_videos/laneFindingVideoOutput.mp4	"LaneFindingVideoSmoothed"



## Discussion: 

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail? What could you do to make it more robust?

todo: write this point out...

- why I used this appraoch
- what worked and what not
- it was adapted to test images (Overfits to this problem --> may not generalize very well but is a good starting point for further work and harder problems)
- ........
- limitations
- One could additionally apply ROI on the warped (i.e. birds eyes view) image as there are still some artefacts due to the damaged road segments or shadow effects. Applying ROI 






