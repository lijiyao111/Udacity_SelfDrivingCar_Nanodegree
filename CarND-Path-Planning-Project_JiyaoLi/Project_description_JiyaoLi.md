# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

## Author
Jiyao Li

## Overall reflection

Motion planning is a very important and big component of self-driving car. To be able to drive the car automatically, safely and smoothly in the real world, the path planning algorithm definitely won't be simple. In Udaicity class, I learned all the general theories of motion planning. Also I learned that there could be different motion planning algorithms for different situations. 

In the class project assignment, we are designing a motion planning algorithm for high way driving in the simulator. First, high way motion planning is easier than urban driving since the traffic pattern is simpler and no pedestrians and bicyclists. Second, this simulator is also a simiplified version of the real-world environment. But it is still a very good practice build the motion planning code from scratch and watch the car drive safely and smoothly.


## Prediction

Prediction is to predict the future trajectory of the other vehicles, to help prevent our vehicle from collision. In the class, model-based and data-based approaches are described. In the class project, I noticed that the other cars mostly drive straight. Thus the prediction logic is as follows:

1. predict the other cars' next position, assuming the car is staying in its lane. 
2. With the car' predicted next position, we predict whether our front path, left path and right path are clear.
  1. if there is no car in front of us within certain distance, the front path is clear.
  2. if there is no car on left of us within certain distance, the left path is clear. 
  3. if there is no car on right of us within certain distance, the right path is clear.


## Behavior planning

With the prediction of other cars, we make high level behavior planning for our car, in terms of the goal position and reference velocity.

1. if front path is not clear:
  1. if left and right path are clear, choose the one that is more clear, i.e. no car in front, or the car is farther away.
  2. if one of the neighboring lane is clear, choose that one. 
  3. no neighboring lane is clear, stay in current lane, but adjust speed accordingly. 
2. if front path is clear:
  1. if we are not in middle lane, and center lane is very clear, switch to the center lane. 
  2. adjust the speed to drive fast if possible. 


## Trajectory generation

Trajectory generation is to generate a smooth and drivable path.

### Generate a smooth path
We take last two points from the previous path, and the goal position of the car, to make a smooth path using spline interpolation. 

### Generate a drivable path
With the reference speed and the bicycle model, we make the future way points on 0.02s time interval. 

