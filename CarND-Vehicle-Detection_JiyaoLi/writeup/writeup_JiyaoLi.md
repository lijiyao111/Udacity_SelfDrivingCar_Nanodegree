## Writeup

---

**Note**, some codes are modified based on code samples from Udacity class. I included two Jupyter notebooks for this project, FeatureExplore.ipynb and VehicleDetection.ipynb. Notebook VehicleDetection.ipynb contains majority of codes for this project while VehicleDetection.ipynb is used to explore the optimal feature set for the model training. 

The structure of the writeup template has be slightly modified. 

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[imageCarSample]: ./carSamples.png
[imageNotCar]: ./notCarSamples.png
[imageFeatures]: ./FeatureMaps.png
[imageWindowM1]: ./windows_method1.png
[imageDetectionM1]: ./detections_method1.png
[imageWindowM2]: ./windows_method2.png
[imageDetectionM2]: ./detections_method2.png
[imageHeat]: ./heatMap.png
[imageOnePipe]: ./oneImage_pipeLine.png
[imageSixPipe]: ./sixImage_pipeLine.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here are some image samples of the `vehicle` and `non-vehicle` classes:

![alt text][imageCarSample]

![alt text][imageNotCar]

The code to extract the HOG, color histogram and spatial features are in cell 7 in FeatureExplore.ipynb. Spatial features are extracted by just resizing the image to make it smaller, in `bin_spatial` function. Color histogram features are extracted using the `color_hist` function. HOG features are extracted using the `get_hog_features` function. Internally, it uses the `skimage.feature.hog` function. In the function signature below, setting the `visualise` to `True` will return a HOG image for display and setting the `feature_vector` to `True` will flatten the returned HOG features to be 1D array. 

```python
hog(image, orientations=orient, 
		pixels_per_cell=(pix_per_cell, pix_per_cell),
		cells_per_block=(cell_per_block, cell_per_block), 
		block_norm = 'L1',
		transform_sqrt=False, 
		visualise=True, feature_vector=True)
```

I then picked 2 car sample images and 2 notCar sample images to visualize the different features (Cell 8 in FeatureExplore.ipynb). I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). The exploring process of different parameters is not shown in the notebook. 

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][imageFeatures]

#### 2. Explain how you settled on your final choice of HOG parameters and other feature parmaters.

Various features will be extracted from the image and used to train model to predict whether a new image is a car or not. The goal is the find the features that can make accurate prediction and also use less time to train model and make prediction. 

Here are various combinations of parameters to extract and use the image features. 

|parameter set | colorspace | orient | pix per cell | cell per block | hog channel | spatial_size | hist_bins | use spatial | use color histogram | use hog | feature normalization | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 |'YUV'| 11| 16| 2| 'ALL'| (32, 32)| 32| False| False| True| False|
| 2 |'YUV'| 11| 16| 2| 'ALL'| (32, 32)| 32| False| False| True| True|
| 3 |'YUV'| 11| 16| 2| 'ALL'| (32, 32)| 32| False| True| True|True|
| 4 |'YUV'| 11| 16| 2| 'ALL'| (32, 32)| 32| True| True| True|True|
| 5 |'YUV'| 11| 16| 2| 'ALL'| (16, 16)| 32| True| True| True|True|
| 6 |'RGB'| 11| 16| 2| 'ALL'| (32, 32)| 32| False| False|  True|False|
| 7 |'HSV'| 11| 16| 2| 'ALL'| (32, 32)| 32| False| False|  True|False|
| 8 |'HLS'| 11| 16| 2| 'ALL'| (32, 32)| 32| False| False|  True|False|
| 9 |'YUV'| 11| 16| 2| 0| (32, 32)| 32| False| False|  True|False|
| 10 |'YUV'| 8| 8| 2| 'ALL'| (32, 32)| 32| False| False|  True|False|

**Note**, "feature normalization" means whether to normalized all the features by 
```python
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
```

From my tests, if using only one feature set, e.g. HOG only, applying feature normalization will reduce the testing accuracy after training model. And later when apply the predictor to sliding window car detection, the result looks bad. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using different parameter sets as shown in the previous table (cell 13 in FeatureExplore.ipynb). Each training time and testing accuracy as shown in the talbe below:

| parameter set | Test accuracy | Training time |
|:---:|:---:|:---:|
|1| 98.65| 0.62|
|2| 98.51| 1.4|
|3| 98.9| 1.3|
|4| 99.18| 6.33|
|5| 99.32| 2.27|
|6| 96.62| 2.05|
|7| 98.45| 0.89|
|8| 98.06| 0.87|
|9| 95.41| 0.47|
|10| 98.73| 3.09|

From this, we can see that parameter set 5 gives the highest testing accuracy and relatively short training time. Thus this parameter set should be chosen for sliding window car detection. (Actually, in VehicleDetection.ipynb, parameter set 1 is used. I tried several different parameter set for vehicle detection in image pipeline and video pipeline. Combining with all the subsequent processing steps, model trained using parameter set 1 is the best. Model trained using parameter set 5 gives more false alarm.)


### Sliding Window Search

I tried two different method to extract features from sliding windows in order to make prediction. Both methods are taught in Udacity class. 

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

There are two methods I used. For both method, we can set xstart, xstop, ystart, ystop to restrict to regions for car detection. In my implementation later, only lower half image is searched. 

In method 1, fixed size sliding windows are used. Windows slide horizontally, then move upward to traverse through horizontally again. Windows overlap with each other for some predefined ratio. In each window, features will be extracted independently. 

In method 2, for better performance, HOG subsampling method is used. First we define what section to search. Then this section will be resized to make the y scale the same as the training image size. Then all HOG features will be calculated. After that we run a sliding windows to collect the extracted features. This process can be applied several times using different sections and the window will overlap with each other. Method 2 also has the advantage to use large window to search for near regions and small search window for far regions. 

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

See the following discussions. To optimize performance, I finally use the HOG sub-sampling method. 

##### 3.1 Method 1, fixed size sliding windows

Cell 11 to 13 in VehicleDetection.ipynb. 

Searched windows:
![alt text][imageWindowM1]

Car detections:
![alt text][imageDetectionM1]


##### 3.2 Method 2, HOG sub-sampling

Cell 14 to 15 in VehicleDetection.ipynb. 

Searched windows:
![alt text][imageWindowM2]

Car detections:
![alt text][imageDetectionM2]


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used a method to use HeatMap to increase the likelyhood for car detections. I recorded the positions of positive detections in each image.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. Codes are in cell 17 in VehicleDetection.ipynb.  

Here's an example result showing the heatmap of several detections from one image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the image:

![alt text][imageHeat]


### Image processing pipeline

Combining all the steps above, I created a pipeline for car detection in each image. Codes are in cell 18 in VehicleDetection.ipynb. 

Here is the result from the image processing pipeline applied on one sample image:

![alt text][imageOnePipe]

Here are results from six sample images. Cars can be detected correctly. And no false positive exist on these six images.

![alt text][imageSixPipe]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Video processing pipeline is in cell 23 in VehicleDetection.ipynb. This is slightly different from the image processing pipeline. Detections are stored across N frames for heatmap to generate. The purpose of using last N video frames is to reduce the variance in vehicle detection and minimize false detections. 

Output of the two testing videos are "test_video_out.mp4" and "project_video_out.mp4". In the processed video, cars can be correctly detectly most of the time and false alarm are very rare. 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I am ok with the results from the two testing videos. Some possible improvements could be

1. Use more powerful neural net model to predict the car. In this way, we can just use the image pixels as input without further extract features. 
2. Allow horizontal sliding window overlapping in the Method 2 of sliding window method. 
3. Think about better way to use multiple video frames to further improve the smoothness of vehicle detection and reduce false detections. 

