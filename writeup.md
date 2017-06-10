#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/model.png "Model Architecture"
[image2]: ./images/center_lane_driving.jpg "Center Lane Driving"
[image3]: ./images/recovery_driving_start.jpg "Recovery Image"
[image4]: ./images/recovery_driving_mid.jpg "Recovery Image"
[image5]: ./images/recovery_driving_end.jpg "Recovery Image"
[image6]: ./images/flipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 78-85) 

The model includes RELU layers to introduce nonlinearity (code line 78), and the data is normalized in the model using a Keras lambda layer (code line 75). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 80-91). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 100). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer and the learning rate was hand tuned to 0.00025 (model.py line 95). I find this value allows me to effectively train the whole model architecture including the dropout layers.

####4. Appropriate training data

In the beginning, I used data collected from center lane driving and recovery driving; however, due to my bad driving skill I wasn't able to produce good enough training data for my behavior cloning model. Finally, I decided to use the sample data set provided by the class. The sample data set consist of good enough data for the model.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to stack layers of neural nets to capture the complexities within the training data that can help us lead to accurately predicting the correct steering angles from the image data.

My first step was to use a convolution neural network model similar to the LeNet architecture. I thought this model might be appropriate because it has a high accuracy in predicting the traffic signs by inputing only the raw images.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it includes dropout layers so I can introduce noise within the hidden layers so the model can generalize to the validation set.

Then I decided to follow the preprocessing and augmentation steps provided by the class to improve the features that are fed into the model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I do not fault this inability to drive through the whole track on the dataset. Rather then adding new data to the dataset, I decided to use the more complex model architecture provided by Nvidia to learn these more difficult cases more appropriately.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 73-93) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                     |     
|:-------------------------:|    
| Input                     |
| Lambda                    |
| Cropping2D                |
| Convolution 5x5, Strides 2|
| RELU                      |
| Convolution 5x5, Strides 2|
| RELU                      |
| DROPOUT                   |
| Convolution 5x5, Strides 2|
| RELU                      |
| DROPOUT                   |
| Convolution 5x5, Strides 1|
| RELU                      |
| DROPOUT                   |
| Convolution 5x5, Strides 1|
| RELU                      |
| DROPOUT                   |
| FLATTEN                   |
| DENSE                     |
| DROPOUT                   |
| DENSE                     |
| DROPOUT                   |
| DENSE                     |
| DENSE                     |

Tere's an image of the final model architecture

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to maneuver out of a risky situation when met with one. These images show what a recovery looks like starting from the vehicle sliding to the right side, recovering back a bit, and finally recovering back to the center of the lane.

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

However, since my driving skill isn't too great, I couldn't create a very good dataset of center lane driving and recovery laps; therefore, I decided to use the Sample Dataset given by the Udacity team which consistd of data representing good driving behaviors.

To augment the data sat, I also flipped images and angles thinking that this would prevent the car from biasing toward the left. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image6]

After the collection process, I had 16072 number of data points. By using images from the left right center camera, I tripled the number of data points I get. I then use a correction factor of 0.2 to correct the stirring angles so better stirring labels are given to the training set. I also cropped the image to a limited window size so that I can ignore most of the unnecesary area for modeling prediction.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the low validation error and the car's ability to drive around the track without giving off to the side.