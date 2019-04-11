# **Behavioral Cloning** 

## Writeup

---

**Note: ** Some codes are modified based on codes from the Udacity coursework. 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture-210x300.png "Model Visualization"
[image2]: ./center.jpg "Center"
[image3]: ./left.jpg "Left"
[image4]: ./right.jpg "Right"
[image5]: ./center_flip.jpg "Center flipped"
[image6]: ./center_sideRoad.jpg "Failed side road"
[image7]: ./loss.png "Loss"
[image8]: ./loss_withDropouts.png "Loss with dropouts"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4  A video recording of my vehicle driving autonomously on the first track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The submission code basically contains four parts:
1. Function to get all training dataset image names and the steering angles. 
2. Generator function to read in the image data during training in batch
3. Build Nvidia CNN model
4. Train model 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The overall strategy for deriving a model architecture was to keep a car in the road when it drive. (For track 2, car also need to be in the right lane. But I did not try track 2.) 

I first tried a simple one CNN layer + one fully connected layer model and then LeNet model. But in my tests, these models can not train the car to finish the entire lap. It could be true that if I spend more time to tune the LeNet model, maybe my car could finish a full lap in track 1. 

Then I used a more powerful model, Nvidia neural net model. After a few tests, model trainied using the architecture drive the car well for a couple of laps. 

Here is the Nvidia model:

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

1. I used the same Nvidia architecture, so there is no dropout layers and max pooling layers. 

**Response to reviewer feedback from the first submission**
The reviewer request me to try using dropout layers and check how the newly trained model work. I added two dropout layers after the first two dense layers. I did not add the third dropout layer after the third dense layer since it only has 10 nodes. 

```python
    model.add(Flatten())
    model.add(Dense(100))
    # Added Dropout layer
    model.add(Dropout(0.40))
    model.add(Dense(50))
    # Added Dropout layer
    model.add(Dropout(0.40))
    model.add(Dense(10))
    model.add(Dense(1))
```

Please see the last section for disccusion on the newly trained model. 

2. The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 97-103). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

3. In some situations, overfitting happens because model is too complicated for the input data, i.e. not enough data to train. Sometimes, giving more training data could reduce the overfitting. In my test, when I only use center camera image, car can not finish full lap. But when I inlcuded center, left, right camera images, car drove well. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 110).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Training data were collected by running the simulator mannualy to drive the car forward for one lap and backward for one lap. I only tested on the first track. 

I used a combination of center lane driving, recovering from the left and right sides of the road. Here are images examples from the center, left, right camera (displayed in the same order) at the same recording time:

![alt text][image2]
![alt text][image3]
![alt text][image4]


To augment the data sat, I also flipped images and angles thinking that this would provide more training data without actually manually driving the car. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image5]

**Note** from my tests, using an Nvidia neural net model and only the center image, car can not finish the full lap. Very interestingly, my car left the main paved road and ran into the side unpaved read. Then it drove for a while and hit the side wall. on the unpaved road, there is no lane line and the road color is very different. I guess it will be much harder to train a car to drive on the unpaved road using camera images. 

Here is where my car took the side road when the model is trained only using the center camera images. 
![alt text][image6]


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I have described in the previous section that I first tried simple CNN architecture and LetNet model. But they don't work well even if I tune the architectures and parameters. Then I tried Nvidia model. The input data using center, left, right camera images. Preprocessing was done by nomalizing the image values and cropping off the top and bottom part. I splitted dataset into 80% training data and 20% validation data. Model was trained after 5 epochs. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input (after cropping)| 90x320x3 RGB image                            | 
| Convolution 5x5       | 2x2 stride, same padding, outputs 43x158x24   |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, same padding, outputs 20x77x36    |
| RELU                  |                                               |
| Convolution 5x5       | 2x2 stride, same padding, outputs 8x37x48     |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, same padding, outputs 6x35x64     |
| RELU                  |                                               |
| Convolution 3x3       | 1x1 stride, same padding, outputs 4x33x64     |
| RELU                  |                                               |
| Flatten               | outputs 8448                                  |
| Fully connected       | outputs 100                                   |
| Fully connected       | outputs 50                                    |
| Fully connected       | outputs 10                                    |
| Fully connected       | outputs 1                                     |


This function in my code (model.py lines 70-90) create the Nvidia model, including the preprocessing steps (line 76-78).

```python
def nVidiaModel():
    """
    Creates nVidea Autonomous Car Group model
    """
    model = Sequential()
    # Normalize image
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop image to only see road
    model.add(Cropping2D(cropping=((50,20), (0,0))))

    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model
```    

#### 3. Creation of the Training Set & Training Process


In total, there are 24108 images, including center, left, right camera images. Each image has size `160 x 320 x 3`. I then preprocessed this data by normalization the image values from range of (0, 255) to range of (-0.5, 0.5). And I removed redundant information in the image by cropping the top and bottom part. After cropping 50 pts from top and 20 pts from bottom, each image size becomes '90x320x3'. In total, there are `24108 x 90 x 320 x 3 = 2 billion` data points as input. (**Note** Normalization and cropping could be down before passing image data into keras model. However, Keras has a convinient `Cropping2D` function to do the job faster. Thus image cropping is done in Keras model. )  

I finally randomly shuffled the data set and put 20% of the data into a validation set.  

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.

I ran the training with 5 epochs. At Epoch 5, it seems that the training and validation errors are still going down. It may be even better to run more epoches until the training and validation errors start to stablize. However, after 5 epochs, the car can already drive well autonomously. Thus I did not try using more epoches. Model was trained on my local PC with GPU GTX 1070, so it took about two minutes to finish trainining. 

Here is the training and validation error plot:

![alt text][image7]

**Response to reviewer feedback from the first submission**

After adding two dropout layers, the training and validation errors are:

![alt text][image8]

Actually, I did not see the validation mse error reduced significantly using the dropout layers. 

Previously I have done [some study](https://github.com/lijiyao111/ConvolutionalNeuralNet_fromScratch/blob/master/Dropout.ipynb
) to better understand the effect the dropout layers. My tests shows that the dropouts does not improve the training accuracy but does improve the validation accuracy. But I saw the improvement on validation accury when the fitting error stablizes after several training epoches. From the error plots, it shows that after 5 epoches, the model error is still reducing. Maybe after more training epoches, we can see the improvement on validation error using dropout layers. 

I used the new model with droput layers to drive the car and saved video as "run2.mp4". Comparing with previous recorded autonomous driving video, "run2.mp4". I kind of feel that the new model drive the car slightly smoother. The previous model without the dropout layers can not drive the car smoothly so I guess the improvement is not significant. Maybe if I tried the second track, the difference will be easily noticeble. I will try the second track when I have more time...... 

