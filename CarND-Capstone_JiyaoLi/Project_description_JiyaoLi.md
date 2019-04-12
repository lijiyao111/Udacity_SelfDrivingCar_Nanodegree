# CarND-Capstone-Project
Self-Driving Car Engineer Nanodegree Program

## Individual Submission
### Author
Jiyao Li
Udacity account email: jiyao.lee@gmail.com

## Overall Reflections
This is a very interesting and challenging project. First I need to implement several different components, e.g. waypoint updater, controller, traffic light detector. And I need to integrate all of the compoments together using ROS, for the car to drive properly in the simulator and testing track. 

Since the real-world autonomous driving is so complicated that a single class project can not incorporate everything. This project is simplified in many ways. For example, the dynamic path planning I learned in the path planning project is not implemented here. Here there is no other vehicles and the path position is pre-loaded from saved files, e.g. sim_waypoints.csv and churchlot_with_cars.csv in the data folder. And for the perception part, only traffic light detection is implemented. In addition, localization and tracking are not needed in this project but definitely important in the real-world situation. 

## Waypoint Updater
First, the waypoint loader loads the pre-definited waypoints for the car trajectory. If there is no red_light detected, the reference velocity at each waypoint is the given target velocity definied in the launch file in the waypoint loader. Once upcoming red light is detected, the velocity of the waypoints will be adjusted so that the car can come to a safe stop before the red light. 


## Drive-By-Wire Node

Twist Controller publishes throttle, brake, and steering to control the car via Drive-by-wire system. Throttle, i.e. acceleration is controlled using PID controller, given a target velocity at each waypoint. Steering is calculated using YawController which simply calculates needed angle to keep needed velocity. Brake is calculated using a simple logic: 1. apply strong brake to stop the car if required and the current velocity is very small. 2. decelerate based on required torque. 

## Traffic Light Detection / Classification

The traffic light detection is implemented using TensorFlow Object Detection API (https://github.com/tensorflow/models/tree/master/research/object_detection), which detect and classify the traffic light using one neural net model. I used the pretrained model provided in the tensorflow repository, "ssd_mobilenet_v1_coco_sim", and retrain the weights using the labeled traffic light images from this project. 

First, I got the camera images via saving the image_color streaming data while running simulator or rosbag, using this command, "rosrun image_view extract_images". Then in order to use the TensorFlow Oject Dection API, images need to be labeled to give both the label, i.e. Red, Green, Yellow, Unknow, and the trafic light position, i.e. bounding box of the traffic light. Two models are training independently for simulation images and track run images. After training and testing, the models are used in the traffic light detection. 


## Reference
** Some algorithms are following the Udacity capsone project walkthrough. And the traffic light detection/classification is using similar pipeline described in this online blog: Self Driving Vehicles: Traffic Light Detection and Classification with TensorFlow Object Detection API(https://becominghuman.ai/traffic-light-detection-tensorflow-api-c75fdbadac62) **

