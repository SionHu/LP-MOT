# LP-MOT
Real-time Low-Power UAV Multi-Object Tracking with 3D Localization

## News
 - (03/13/2021) Initializeed this GitHub repo

## Overview
We popose a method to achivee real-time multi-object tracking on UAV with 3D localizations of the tracked objects.

## Info about odometry branch
This branch stores the work using FastMOT and drone sensor data to estimate the distance from the drone to the tracked objects, then estimate their GPS location.
### Todo:
- Implement calculations for objects in different regions in the frame.
- Add numba to all the calculations. It currently slows everything down.
