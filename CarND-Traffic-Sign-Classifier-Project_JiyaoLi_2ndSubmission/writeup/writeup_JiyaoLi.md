# **Traffic Sign Recognition** 

## Writeup

**Note:** Some codes are modified based on code examples from Udacity class. 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[sampleImage0]: ./Sample_label0.png
[sampleImage1]: ./Sample_label1.png
[sampleImage2]: ./Sample_label2.png
[sampleImage3]: ./Sample_label3.png
[sampleImage4]: ./Sample_label4.png
[sampleImage5]: ./Sample_label5.png
[sampleImage6]: ./Sample_label6.png
[sampleImage7]: ./Sample_label7.png
[sampleImage8]: ./Sample_label8.png
[sampleImage9]: ./Sample_label9.png
[sampleImage10]: ./Sample_label10.png
[sampleImage11]: ./Sample_label11.png
[sampleImage12]: ./Sample_label12.png
[sampleImage13]: ./Sample_label13.png
[sampleImage14]: ./Sample_label14.png
[sampleImage15]: ./Sample_label15.png
[sampleImage16]: ./Sample_label16.png
[sampleImage17]: ./Sample_label17.png
[sampleImage18]: ./Sample_label18.png
[sampleImage19]: ./Sample_label19.png
[sampleImage20]: ./Sample_label20.png
[sampleImage21]: ./Sample_label21.png
[sampleImage22]: ./Sample_label22.png
[sampleImage23]: ./Sample_label23.png
[sampleImage24]: ./Sample_label24.png
[sampleImage25]: ./Sample_label25.png
[sampleImage26]: ./Sample_label26.png
[sampleImage27]: ./Sample_label27.png
[sampleImage28]: ./Sample_label28.png
[sampleImage29]: ./Sample_label29.png
[sampleImage30]: ./Sample_label30.png
[sampleImage31]: ./Sample_label31.png
[sampleImage32]: ./Sample_label32.png
[sampleImage33]: ./Sample_label33.png
[sampleImage34]: ./Sample_label34.png
[sampleImage35]: ./Sample_label35.png
[sampleImage36]: ./Sample_label36.png
[sampleImage37]: ./Sample_label37.png
[sampleImage38]: ./Sample_label38.png
[sampleImage39]: ./Sample_label39.png
[sampleImage40]: ./Sample_label40.png
[sampleImage41]: ./Sample_label41.png
[sampleImage42]: ./Sample_label42.png
[compPreproc]: ./compPreprocessing.png 
[labelCount]: ./sampleLabelCount.png 
[valAccuracy]: ./valAccuracy.png 
[webConv1]: ./webImageConv1.png 
[webConv2]: ./webImageConv2.png 
[webImageOneSample]: ./webImageSamplePrep.png 
[webImagesPred]: ./webImagesPred.png 
[leNetImage]: ./leNetImage.jpg 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

<!-- You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) -->

My Jupyter notebook code project is also in the submitted compressed zip folder, "Traffic_Sign_Classifier.ipynb". 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x1
* The number of unique classes/labels in the data set is 43




#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of sample from training data set. 15 samples from each traffic sign category are displayed. 

Image samples:
![alt text][sampleImage0]
![alt text][sampleImage1]
![alt text][sampleImage2]
![alt text][sampleImage3]
![alt text][sampleImage4]
![alt text][sampleImage5]
![alt text][sampleImage6]
![alt text][sampleImage7]
![alt text][sampleImage8]
![alt text][sampleImage9]
![alt text][sampleImage10]
![alt text][sampleImage11]
![alt text][sampleImage12]
![alt text][sampleImage13]
![alt text][sampleImage14]
![alt text][sampleImage15]
![alt text][sampleImage16]
![alt text][sampleImage17]
![alt text][sampleImage18]
![alt text][sampleImage19]
![alt text][sampleImage20]
![alt text][sampleImage21]
![alt text][sampleImage22]
![alt text][sampleImage23]
![alt text][sampleImage24]
![alt text][sampleImage25]
![alt text][sampleImage26]
![alt text][sampleImage27]
![alt text][sampleImage28]
![alt text][sampleImage29]
![alt text][sampleImage30]
![alt text][sampleImage31]
![alt text][sampleImage32]
![alt text][sampleImage33]
![alt text][sampleImage34]
![alt text][sampleImage35]
![alt text][sampleImage36]
![alt text][sampleImage37]
![alt text][sampleImage38]
![alt text][sampleImage39]
![alt text][sampleImage40]
![alt text][sampleImage41]
![alt text][sampleImage42]


Here is the histogram showing the count of images in each traffic sign category, from training dataset, validation dataset and testing dataset. 

![alt text][labelCount]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

My preprocessing steps are converting RGB images to grayscale and normalization. 

Converting to grayscale is to reduce parameter number in the neural net model, especially for the first layer. For example, assuming a fully connnected first layer, if the input is colorful image, each weight filter need to have size of `32*32*3`. If input image is grayscale, each weight filter reduces size to `32*32*1`. This could potentially reduce the overfitting risk. And I feel that all the traffic signs look quite different in grayscale, making it very likely to be distinguished just using grayscale. 

The reason for normalizing the image is that, we're going to be multiplying (weights) and adding to (biases) these initial inputs in order to cause activations that we then backpropogate with the gradients to train the model. We'd like in this process for each feature to have a similar range so that our gradients don't go out of control (and that we only need one global learning rate multiplier).

From practical point of view, without normalization, it will be super difficult to train model since each image may have different range in absolute value. 


Here is an example of a traffic sign image after grayscaling and normalization, comparing with the orignal colorful image:

![alt text][compPreproc]


I did not create any augmented image dataset.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


|Layer | Description |
|:---------------------:|:---------------------------------------------:| 
|Input | RGB image 32x32x1|
|Convolutional Layer 1 | 1x1 strides, valid padding, Outputs 28x28x16|
|RELU|  |
|Max Pool| 2x2 pooling, Outputs 14x14x16|
|Convolutional Layer 2 | 1x1 strides, valid padding, Outputs 10x10x64|
|RELU|  |
|Max Pool | 2x2 pooling, Outputs 5x5x64|
|Fatten| To connect to fully-connected layers |
|Fully-connected Layer 1| Outputs 1600|
|RELU|  |
|Dropout| keep 60% nodes |
|Fully-connected Layer 2| Outputs 240
|RELU|  |
|Dropout| keep 60% nodes |
|Fully-connected Layer 3| Outputs 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the [LetNet-5](http://yann.lecun.com/exdb/lenet/) Model introduced by Yann Lecun. This model is also used in the Udacity class. I did not change the structure of the LetNet Model, except that two dropout layers were added after the first two fully connected layers. Here is a good [Introduction for different Neural net models](https://adeshpande3.github.io/The-9-Deep-Learning-Papers-You-Need-To-Know-About.html). As described in the Udacity class as well, there are more complex and powerful Neural network models. But LetNet model still has its popularity due to its simple structure and performance. And LetNet model was built for image recognition fo hand written digits, similar to the goal for this project. These are the reasons to build my model based on LetNet model in my project. 

Here is the original LetNet Model structure:

![alt text][leNetImage]

However, using just the LeNet model provided by Udacity, I could not achieve the goal of over 93% accuracy in validation set. Then I tried to add more nodes (i.e. weights) in the covolutional layers and fully connected layers. I also added two dropout layers after the fully connected layers to reduce overfitting. 

In my training, I used 40 epoches and 128 batch size. Starting training rate is 0.01. But since Adam optimizer was used, training rate does not need to be mannually adjusted. I mainly played with the number of Epoches. I found that after 10 epoches, validation accuracy starts to stablize around 96%. It should be ok to stop training then. But after 40 epoches of training, the validation accuray increases to 97%. So I kept the training epoches to be 40 ( And because I used GPU in my PC so trainig is quite fast. :-) ). 

Here is the plot of validation accuracy:

![alt text][valAccuracy]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Reasons are explained in details in the previous section. 

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.1% 
* test set accuracy of 95.2%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I downloaded 34 German traffic signs from the website. They all look similar to the image dataset provided by Udacity. However, their image size are not `32*32*3`. Thus, first processing step is to resize them to be the sample image size as the same size of the Udacity provied images. 

I first thought accurate predictions on some of these images could be difficult because of the their image quality. For example, some images, e.g. "00037.ppm", looks very dark. And "00045.ppm" is difficult for me to identify as bicyle passing sign after image resizing. 

After resizing and preprocessing these images using the same procedures as used before model training, I used the previoulsy trained model to make predictions on these 34 images. 

Here are the 34 web image. Printted on top of these images are their file name, true category and predicted categories:

![alt text][webImagesPred] 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The results of the prediction are already shown in the previous image. 

The model was able to correctly guess 34 of the 34 traffic signs, which gives an accuracy of 100%. This is kind of to my surprise. But I think since the sample size is relatively small, only 34 images, even the testing accuracy is 95%, it could very likely be true that the predictions on these images are correct.  

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

All top 5 softmax probabilities are shown for 34 web images. The detailed results can be found in the 31st cell of the Ipython notebook.

Here I just choose results for two images to discuss. 

For the first image, the model is quite sure that this is a Speed limit of (100km/h) (probability of 1.0). Not surprisingly, model also thinks this image could be speed limit sign with other speed numbers, altough the probability is much lower than the top prediction.  

| Image name | 00047.ppm |
| True label | Speed limit (100km/h) |
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
|   1.000000 | 7 - Speed limit (100km/h) |
|   0.000000 | 40 - Roundabout mandatory |
|   0.000000 | 5 - Speed limit (80km/h) |
|  0.000000 | 8 - Speed limit (120km/h)|
|  0.000000 | 12 - Priority road | 

For the second image, the highest probability is reduced but still very high. And we can see that probabilities for second and third predictions are higher compared with the situation in the previous image. 

| Image name | 00037.ppm |
| True label | Beware of ice/snow |
| Probability         	|     Prediction |	        					| 
|:---------------------:|:---------------------------------------------:| 
|   0.996455 | 30 - Beware of ice/snow |
|   0.001557 | 11 - Right-of-way at the next intersection |
|   0.000961 | 28 - Children crossing |
|   0.000481 | 23 - Slippery road |
|   0.000232 | 20 - Dangerous curve to the right |


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Taking example of one image, I will shown the neural network's weight from the first and second convolutional layers. Based on the weights from the first convolutional layer, we can see that each weight filter focus on some specific regions and edges of the input images. Weights from the second layer is not that intuitive. But it seems that different weights are focusing on different regions as well.   

Here is the image after preprocessing:

![alt text][webImageOneSample] 

Here is the weight images from the first Convolutional layer:
![alt text][webConv1]

Here is the weight images from the first Convolutional layer:
![alt text][webConv2]