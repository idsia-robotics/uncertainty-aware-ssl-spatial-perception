# Uncertainty-Aware Self-Supervised Learning <br> of Spatial Perception Tasks

*Mirko Nava, Antonio Paolillo, Jerome Guzzi, Luca Maria Gambardella, and Alessandro Giusti*

Dalle Molle Institute for Artificial Intelligence, USI-SUPSI, Lugano (Switzerland)

### Abstract
We propose a general self-supervised learning approach for spatial perception tasks, such as estimating the pose of an object relative to the robot, from onboard sensor readings.
The model is learned from training episodes, by relying on: a continuous state estimate, possibly inaccurate and affected by odometry drift; and a detector, that sporadically provides supervision about the target pose.  We demonstrate the general approach in three different concrete scenarios: a simulated robot arm that visually estimates the pose of an object of interest; a small differential drive robot using 7 infrared sensors to localize a nearby wall; an omnidirectional mobile robot that localizes itself in an environment from camera images. Quantitative results show that the approach works well in all three scenarios, and that explicitly accounting for uncertainty yields statistically significant performance improvements.

### Video
[![Uncertainty-Aware Self-Supervised Learning of Spatial Perception Tasks](https://github.com/idsia-robotics/uncertainty-aware-ssl-spatial-perception/blob/main/video/video.gif)](https://youtu.be/G3cIDRrkfZY)
