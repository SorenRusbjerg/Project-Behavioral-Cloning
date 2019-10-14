# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/Center.jpg "Center driving"
[image3]: ./examples/Recovery1.jpg "Recovery Image"
[image4]: ./examples/Recovery2.jpg "Recovery Image"
[image5]: ./examples/Recovery3.jpg "Recovery Image"
[image6]: ./examples/TrackTwo.jpg "Normal Image"
[image7]: ./examples/croppedImage.png "Cropped Images"
[image8]: ./examples/TrainHistory.png "Training loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture

#### 1. An appropriate model architecture has been employed

My model consists of a input layers, convolutional layers and dense layers.

**Input layers**
 Input is first cropped top/bottom to suiteable size without car front and top landscape. The input is then normalized and resized to 128x128 using a Keras lambda layer, to decrease the image resolution before conv. layers.

**Convolutional layers**
The model uses a convolution neural network with 3 convolutional layers with 5x5 and 3x3 filter sizes and depths at 8, 32 and 64 featuremaps, with maxpooling layers in between to reduce resolution.    

The model includes RELU layers to introduce nonlinearity, 

**Dense layers**
The model ends in three dense layers of 1024, 256 and 12 neurons.  

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting using 50% dropout. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. 


#### 3. Model parameter tuning

The model used an adam optimizer, with a learning rate lower than default (lr=0.0005) to make sure that the loss would keep decreasing while training.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, driving the track backwards, using all three camera images from the car and also using the jungle track. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
The overall strategy for deriving a model architecture was to use a pretrained convolutional net, using transfer learning. MobileNetV2 was found but was not included in used Keras version, so I tried using MobileNet instead. This net should be a fast network trained on *imagenet* data, with good usage for robots, vehicles etc. 

I however was not able to get it to improve the validation loss using this, and after trying using different tuning of hyperparameters, without luck (the vehicle almost only went in a straight line) I dropped using this net. 

Instead I made my own convolution neural network model similar to the as described earlier. This network having more conv layers I imagined was powerful enough to do the job of controlling the car steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. It seemed that this network, which employed dropoutlayer was able to reduce both training and validation loss. 

I however wanted to get an even better network, by trying to use batch normalization layers in the between conv. layers. This I however was not able to get working, and reducing validation loss. So I skipped these layers again. 

In between these steps I ran the simulator to see how well the car was driving around track one. I found that my own network was able to drive sensible, and decided to go with that. There were a few spots where the vehicle fell off the track, especially in sharp turns. To improve the driving behavior in these cases, I recorded more training data of the specific turns, and also data of recovery driving where I started in the side and ran to the middle of the road. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-114)  is visualized below:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. I then ran the track backwards. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from the road sides These images show what a recovery looks like starting from a right side recovery:

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points:

![alt text][image6]

The input images was cropped by the N.N. to get better generalization (left: input; right: cropped):

![alt text][image7]


After the collection process, I had app. 53.500 number of data points. 

I finally randomly shuffled the data set and put 20% of the data into a validation set. 


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was in the range of 3 to 7 as evidenced by when the validation loss started to flatten. I retrained the network a couple of times when I got new turning data, and therefore the total number of epochs became higher. 

I used an adam optimizer so that manually training the learning rate wasn't necessary, and as it is an effective optimizer. An example of my training process can be seen below:

![alt text][image8]
