# **Finding Lane Lines on the Road** 

## Final Writeup

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # "Image References"

[image1]: ./pipe_images_output/gray.jpg "grayscale"
[image2]: ./pipe_images_output/blur.jpg "gaussian blur"
[image3]: ./pipe_images_output/canny.jpg "canny edges"
[image4]: ./pipe_images_output/mask.jpg "region select"
[image5]: ./test_images_output/solidWhiteCurve.jpg "hough transform"

---

### Reflection

### 1. Pipeline Description

My pipeline consisted of the following steps:

1.  Convert the image to *grayscale* to make it easier to work with. 

  ![alt text][image1]

2.  Apply *Gaussian Smoothing* to the image. I chose to use a *kernel size* of *7* for this step as I found that yielded the best results in edge detection step and minimized the number of false positives

  `kernel_size = 7`

  ![alt text][image2]

3.  Perform *Canny edge detection* on the image to obtain an image to which the Hough transform can be applied.  I decided to use the following thresholds as I  found that these threshold values yield the best lane detection results on the test image.

    `high_threshold = 100`

   `low_threshold = 50`

  ![alt text][image3]

4.  Apply a mask to the image to select the region of interest. This step in the pipeline proved to be very useful for minimizing the detection of false positives such as other cars and the road shoulder. I applied a polygon mask using the following vertices to the image

  `vertices = np.array([[(xsize-400,ysize-200),(400, ysize-200), (0,ysize), (xsize,ysize)]])`

  ![alt text][image4]


5.  The next and final step in pipeline involved applying the *Hough transform* to the pre-processed image. After some experimentation I found that the following parameters yielded the best results:
      `**rho = 1**`
      `**theta = np.pi/180**`
      `**threshold = 50**`
      `**min_len = 30**`
      `**max_gap = 50**` 

      I added the houghlines output image to a copy of the original image using the  addweighted routine

      `line_edges = cv2.addWeighted(hough_image, 1, line_image, 1, 0)` 

![alt text][image5]

In order to extend the detected lane lines I modified the draw lines routine to use the *slope* and *y-intercept* of the detected lines to make them continuous and smooth. See code below. 

    xsize = img.shape[1]
    ysize = img.shape[0]
    for line in lines:
        for x1,y1,x2,y2 in line:
            #Compute slope and y-intercept
            m = float(y2-y1)/(x2-x1)
            c = y2 - m*x2
            slope_thresh = 0.5
            #Compute new start and end points at bottom and top of ROI
            if abs(m) > slope_thresh:
                y1 = ysize
                x1 = int((y1 - c) / m) 
                y2 = ysize - 200
                x2 = int((y2 - c) / m)
                #Draw lines
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)
I used a slope threshold of *0.5* to eliminate the horizontal lines seen in the challenge video from the front of the car and the road.

 `slope_thresh = 0.5`

        #Compute new start and end points at bottom and top of ROI
        if abs(m) > slope_thresh:
### 2. Identify potential shortcomings with your current pipeline

1. The pipeline processing steps are highly dependent on the image quality such as lighting conditions and road conditions. While it works well on the test images it may perform poorly under darker/ rainy conditions.
2. While this pipeline transform works well on straight roads and mild curves it might have trouble with sharper curves and inclines


### 3. Suggest possible improvements to your pipeline

1. An improvement might include adding adaptation to the pipeline to compensate for varying lighting, road ad driving conditions. 
2.  Also using a larger test image set to fine tune the parameters on a wider set of road and environmental conditions would add further robustness to the algorithm.