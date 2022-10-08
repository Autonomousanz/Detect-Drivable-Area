# Convolutional Neural Network for Drivable Area Segmentation

Deep Learning Project: Drivable Area Segmentation for poor weather conditions

**Team :** Pranava Swaroopa, Sanskruti Jadhav , Siddharth Joshi
{pswaroo, sanskrj, sajoshi}@clemson.edu

**Acknowledgements :** Dr. Rahul Rai and Shengli Xu 

Motivation:

Vehicles are increasingly offering advanced driver assistance in the form of lane keep assist, emergency
brake assist, and many other things to drive safer. Many vehicles like Teslas depend on camera image
feed to understand their environment. Since a camera is cheaper and easier to adapt compared to
other advanced sensors like LiDAR or RADAR. Lane-keep assist systems based on cameras need
a robust drivable area detection algorithm that can consistently perform well in inclement weather
conditions like cloudy, rainy, snowy, and foggy. We use a simple encoder-decoder type convolution
neural network to perform lane detection. We present our results from training on the state-of-the-art
BDD100K dataset. We achieved a mIOU of 44.15% while only using worse weather condition
images

### ***Architecture of CNN ***

![Optimized Model Architecture](https://user-images.githubusercontent.com/64002247/194716904-686d1cf7-de94-4893-a9b6-8e16d0168f55.png)


There are 15 Layers of Neural network with input of 80 X 160 pixel image size , the total layers of following types

5 layers : Convolution
6 layers : DeConvolution
2 layers : Max Pooling
2 layers : Up sampling 


![Optimized Model Parameters](https://user-images.githubusercontent.com/64002247/194716806-b06cd156-db33-4531-b987-cd977d310270.png)


### ***Environment and Dependency***
* Tensorflow 3.5
* OpenCV Library
* Python 3.6
* CUDA 11.0.3 cuDNN 8.0 for Tesla V100 - 16GB
* High Performance Computing Cluster - Palmetto Cluster, Clemson University

### ***Results:***

![Results](https://user-images.githubusercontent.com/64002247/194717008-4e1bd9a2-31e1-45d1-bfe9-5455ef694ffa.png)


![image](https://user-images.githubusercontent.com/64002247/194717032-cd7fcab5-2b41-4290-9b3a-a8adbb6a1503.png)


### Deploying CNN on test Video 
![Result_Demonstration](https://user-images.githubusercontent.com/64002247/194719232-831a2931-d6b6-4725-912c-a1337c773022.gif)



### Future Scope 

1. As observed from the results of the initial model and final optimum model, the feature learning of the
images was improved. However, the improvement is not sufficient which can be observed in the video frame
upon deploying this model for foggy weather test cases. In order to improve the feature learning additional
convolutional layering followed by deconvolution layering can be implemented. Fine tuning of the model is
required to reduce the disparity between predicted and ground truth.

2. The model needs more tuning of hyperparameters for optimum performance, for example the model can be
trained for more epochs with a dual GPU with better learning rate. Since the loss of the model reduces minutely
throughout the 10 epochs, if the learning rate is increased while allowing more repetitions of training(epochs)
a better model output can be achieved.

3. The time taken for model training is one major bottleneck even for small image sizes, we were training the
model on single GPU however using dual GPU power the training time can be reduced for big data size and
thereby using original image size of 720 x 1280 pixels can be then used as an input to the model. Thus a more
complex model with higher dimensions of image, a better feature learning can be done.

We used the network to predict lanes on a video. That way using the same network and adding a Recurrent
Neural Network to make use of the temporal features would yield better results.
