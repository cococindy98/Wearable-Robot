# Wearable-Robot

# A Preliminary Study on Semantic Segmentation Techniques for Environment Recognition of Walking Assistant Robot 

## Research Motivation
There is little data in the public dataset that reflects the geographical characteristics of Korea. Therefore, we are going to obtain a dataset that contains a lot of mountainous terrain. Do a semantic segmentation using datasets that fit the geographical characteristics of Korea.
Furthermore, we would like to perform a real time semantic segmentation that suits the geographical characteristics of Korea.


## Code Description
> * bag_imu.py : rosbag file 
> * bag_rgb_depth.py : bag file data( rgb data, depth data) -> image data 
> * data_name.py : Record image data's name into CSV file
> * head_chest.py : Get data from realsense camera
> * label.txt : Labels I specified (Refered PASCAL label)
> * labelme2voc.py : Convert json file as PASCAL dataset image
> * model : What I used When training
> * train_test.py  : training & test my custom dataset
> * xavier_AP.py : Make Xavier as API


## Result of Study 
![image](https://user-images.githubusercontent.com/83954540/141611218-0ce5595c-36cd-4fa0-a088-8b1939bf1598.png)
![Alt text]

