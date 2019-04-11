# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

<!-- [image1]: ./examples/grayscale.jpg "Grayscale" -->

[image1]: ./test_images2/bridge2.jpeg
[image6]: ./test_images_output/gray_bridge2.jpeg
[image2]: ./test_images_output/pre_bridge2.jpeg
[image3]: ./test_images_output/blur_bridge2.jpeg
[image4]: ./test_images_output/canny_bridge2.jpeg
[image5]: ./test_images_output/final_bridge2.jpeg

---

### Reflection

**Overall reflection**

At the beginning of working on this project, I followed the steps as what was taught in the Udacity class. First converting the colorful image to gray scale, then smooth the image, running canny edge detection, and hough lines detection. Finally I tried to draw two lines from all the straight lines detected in hough transform. 

This pipline works well for the "solidWhiteRight.mp4" image. But for the "challenge.mp4" video, especially when there are shadows on the road and when road color changes from black to gray, lanes can not be detected correctly. No matter how I tune the parameters in smoothing, canny edge detection, hough transform, it still could not work well for the challenge video. 

Thus, improvements were made, mostly by taking the fact that lane line colors are just yellow and white. Preprocessing steps were included to boost the sections with yellow and white colors in the gray image. 

My pipeline could correct draw lane lines in all three videos. 

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. Parameter values in the following steps were tunned to ensure they can produce a good detection on the images and all three videos. 

1. Convert colorful image into gray scale. Then boost up the sections that are in yellow and white color in the orignial image. 
```python
def preprocess(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # define range of color in HSV
    lower_yel = np.array([20,100,100])
    upper_yel = np.array([30,255,255])
    lower_wht = np.array([0,0,235])
    upper_wht = np.array([255,255,255])
    
    # Threshold the HSV image to get only yellow/white 
    yellow_mask = cv2.inRange(hsv_img, lower_yel, upper_yel)
    white_mask = cv2.inRange(hsv_img, lower_wht, upper_wht)

    # Find the region with yellow and white color
    full_mask = (yellow_mask>0) | (white_mask>0)
    
    # boost up the region in gray image which had yellow and white color by 2 times
    # subdue other regions by 1.5 times
    subdued_gray = (gray_img / 1.5).astype('uint8')
    boosted_lanes = np.copy(subdued_gray).astype('uint16')
    boosted_lanes[full_mask] = (boosted_lanes[full_mask] * 3)
    boosted_lanes[boosted_lanes>255] = 255
    
    return boosted_lanes.astype('uint8')
```

2. Gaussian smoothing of image. I set the kernal size as 5. 
3. Canny edge detection to show only the sections with high gradient. Parameters are `low_threshold=10, high_threshold=150`.
4. Hough transform to detect straight lines. Parameters are `rho=3, theta=np.pi/180, threshold=70, min_line_len=70, max_line_gap=250`.
5. Based on all the straight lines detection in step 4, find the possible left and right lane lines. This is done by first finding all the possible lines detected in Step 4 with slope close to either the estimated left lane or estimated right lane. Then left lane or right lane is estimated from the right dipping lines or left dipping lines.

```python
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    positive_m_x = []
    positive_m_y = []
    negative_m_x = []
    negative_m_y = []
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            slop = ((y2-y1)/(x2-x1))
            # find possible right lane lines (left dipping lines)
            if slop < -0.6 and slop > -0.9:
                negative_m_x.extend((x1,x2))
                negative_m_y.extend((y1,y2))
            # find possible left lane lines (right dipping lines)
            elif slop < 0.7 and slop > 0.5 :
                positive_m_x.extend((x1,x2))
                positive_m_y.extend((y1,y2))

    width = img.shape[1]
    height = img.shape[0]

    # fit one straight line to all the right dipping line end points
    if len(positive_m_x) > 0 and len(positive_m_y) > 0:
        m,b = np.polyfit(positive_m_x, positive_m_y, 1)
        cv2.line(img, (0, int(b)), (width, int(width * m + b)), color, thickness)     
    # fit one straight line to all the left dipping line end points
    if len(negative_m_x) > 0 and len(negative_m_y) > 0:
        m,b = np.polyfit(negative_m_x, negative_m_y, 1)
        cv2.line(img, (0, int(b)), (width, int(width * m + b)), color, thickness)       

```


#### 1.1 Example images at different steps in the pipeline:


Original Image:

![Original Image][image1]

After converting to gray scale:

![After converting to gray scale][image6]

Boost up the regions in white and yellow color:

![Boost up the regions in white and yellow color][image2]

After blurring:

![After blurring][image3]

Canny edge detections:

![Canny edge detections][image4]

Draw the detection lane lines:

![Draw the detection lane lines][image5]



### 2. Identify potential shortcomings with your current pipeline

1. As I described in the **Overall Reflection**, if there is no boosting up of the sections in yellow and white color in the image, no matter how I tuned different parameters values, lane lines can not be detected correctly in the "challenge.mp4" video, when there are shadows on the road or at the bridge. After a preprocessing step is included, this is no longer a shortcoming in the current pipeline, unless the road color is very close to white or yellow, which is unlikely based on everyday life experience.  

2. Current algorithm is to identify the lane lines as straight lines. Thus if the lane curvature is large, this lane line detection pipeline may not work well, or will just detect the very beginning of the lane, where it can be approximated as straight line.   


### 3. Suggest possible improvements to your pipeline

The major possible improvement I would suggest is to treat the lane line not just as a straight line, but to consider it either as an continuous straight line or curved line. Actually the fourth project in CarND Term 1 is "Advanced Lane Lines", which does consider the lane lines as curved lines.  
