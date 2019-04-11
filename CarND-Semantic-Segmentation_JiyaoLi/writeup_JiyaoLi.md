

# CarND Sementic Segmentation Project

## Author
Jiyao Li


[//]: # (Image References)

[image1]: ./JARS_13_1_016501_f003.png "FCN Model Visualization"
[image2]: ./tested_images/um_000017.png "Tested image example1"
[image3]: ./tested_images/um_000045.png "Tested image example2"
[image4]: ./training_loss.png "Training loss"

###Model Architecture
**Overall Refections:**
1. 1x1 Convolution layer is mostly for changing dimension. The output will have the same height and weight, but different depth, depending on the number of filter kernels.
2. Deconvolution or upsampling layer is kind of reverse of convolutional layer. It takes each element, multiplies with the kernel weight and add the output together. 
3. Skip layer is taking two layers with the same dimension and element-wise adding then together.
4. The decoding layers are not exact the same as the encoding layers. In our model, it is much simpler than the encoding layers. In our case, the #classes is 2, i.e. free road surface or not.

![alt text][image1]
Figure 1, Architecture of the FCN model (may not be exactly the same as the model used in class project)

###Training loss
I ran training using 24 epochs. Each epoch has 5 batches. That means that the whole training image sets are used in training for 24 times. Each training is only using 5 images.

Below is the training loss plot. After 7 epochs, the loss starts to stablize. 
![alt text][image4]
Figure 2, Training loss

###Testing images
After training, model is used to predict free road surface using test images. In most cases, the free surface can be correctly annotated. Here are two examples:
![alt text][image2]
![alt text][image3]