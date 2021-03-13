# SLAM-using-Extended-Kalman-Filter
Implementation of a 2D EKF-SLAM solver in Python to recover the trajectory of the robot and the positions of the landmarks from the control input and measurements.  
The measurement (observing the surrounding environment and measuring some landmarks) and control (executing a control input to move) steps are repeated several times.  
  
The result:  
In the output figure, the magenta and blue ellipses represent the predicted and updated uncertainties of the robot’s position at each time respectively. The black dots are the ground truth positions of the landmarks. Also, the red and green ellipses represent the initial and all the updated uncertainties of the landmarks, respectively.
![GitHub Logo](/results/result.png)
