#!/usr/bin/env python
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <ros/ros.h>


import rospy
from rostopic import get_topic_type
from sensor_msgs.msg import Image, CompressedImage
from sensor_msgs.msg import PointCloud2,Imu
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import cv2
import numpy as np


class HeadChest(object):
    def __init__(self):
        # Set CvBridge
        self.bridge = CvBridge()
        self.color_img_chest = None
        self.depth_img_chest = None
        self.point_img_chest = None
      
        img_in_topicname1_chest = '/service_vision/camera_chest/color/image_raw'
        img_in_topicname2_chest = '/service_vision/camera_chest/depth/image_rect_raw'
        img_in_topicname3_chest = '/service_vision/camera_chest/depth/color/points'
       
        
        self.sub_rgb_chest = message_filters.Subscriber(img_in_topicname1_chest,
                                        Image,
                                        queue_size=5)
       
        self.sub_depth_chest = message_filters.Subscriber(img_in_topicname2_chest,
                                        Image,
                                        queue_size=5)

        self.sub_point_chest = message_filters.Subscriber(img_in_topicname3_chest,
                                        PointCloud2,
                                        queue_size=5)
                       

        self.ts_chest = message_filters.TimeSynchronizer([self.sub_rgb_chest, self.sub_depth_chest, self.sub_point_chest], 10)
        self.ts_chest.registerCallback(self.img_cb_chest)
        
        img_out_topicname1_chest = '/chest/camera/color/image_raw'
        img_out_topicname2_chest = '/chest/camera/depth/image_rect_raw'
        img_out_topicname3_chest = '/chest/camera/depth/color/points'
        
        self.pub_rgb_chest = rospy.Publisher(img_out_topicname1_chest,
                                   Image,
                                   queue_size=1)
        self.pub_depth_chest = rospy.Publisher(img_out_topicname2_chest,
                                              Image,
                                              queue_size=1)
        self.pub_points_chest = rospy.Publisher(img_out_topicname3_chest,
                                              PointCloud2,
                                              queue_size=1)                                      
       

        # Store last image to process here
        self.color_img = None
        self.depth_img = None
        self.point_img = None
      
        img_in_topicname1 = '/service_vision/camera_head/color/image_raw'
        img_in_topicname2 = '/service_vision/camera_head/depth/image_rect_raw'
        img_in_topicname3 = '/service_vision/camera_head/depth/color/points'
       
        
        self.sub_rgb = message_filters.Subscriber(img_in_topicname1,
                                        Image,
                                        queue_size=5)
       
        self.sub_depth = message_filters.Subscriber(img_in_topicname2,
                                        Image,
                                        queue_size=5)

        self.sub_point = message_filters.Subscriber(img_in_topicname3,
                                        PointCloud2,
                                        queue_size=5)
                       

        self.ts = message_filters.TimeSynchronizer([self.sub_rgb, self.sub_depth, self.sub_point], 10)
        self.ts.registerCallback(self.img_cb)
        
        img_out_topicname1 = '/head/camera/color/image_raw'
        img_out_topicname2 = '/head/camera/depth/image_rect_raw'
        img_out_topicname3 = '/head/camera/depth/color/points'
        
        self.pub_rgb = rospy.Publisher(img_out_topicname1,
                                   Image,
                                   queue_size=1)
        self.pub_depth = rospy.Publisher(img_out_topicname2,
                                              Image,
                                              queue_size=1)
        self.pub_points = rospy.Publisher(img_out_topicname3,
                                              PointCloud2,
                                              queue_size=1)                                      
       
    def img_cb_chest(self,image_chest,depth_chest,points_chest):
        """
        Callback for the Image or Compressed image subscriber, storing
        this last image and setting a flag that the image is new.
        :param Image or CompressedImage image: the data from the topic
        """
        
        self.color_img_chest = image_chest
        self.depth_img_chest = depth_chest
        self.point_img_chest = points_chest
    def img_cb(self, image,depth,points):
        """
        Callback for the Image or Compressed image subscriber, storing
        this last image and setting a flag that the image is new.
        :param Image or CompressedImage image: the data from the topic
        """
        self.color_img = image
        self.depth_img = depth
        self.point_img = points
        
     

    def do_stuff(self):
        """
        Method to do stuff with the last image received.
        First we transform the image message to a cv2 image (numpy.ndarray).
        Then we do OpenCV stuff with it.
        And we publish the new image.
        """

        self.pub_rgb.publish(self.color_img)
        self.pub_depth.publish(self.depth_img)
        self.pub_points.publish(self.point_img)

        self.pub_rgb_chest.publish(self.color_img_chest)
        self.pub_depth_chest.publish(self.depth_img_chest)
        self.pub_points_chest.publish(self.point_img_chest)
       


    def run(self):
        """
        Method to do stuff at a certain rate.
        """
        r = rospy.Rate(1)
        while not rospy.is_shutdown():
            if self.point_img is not None:
                self.do_stuff()
            r.sleep()


if __name__ == '__main__':
    rospy.init_node('camera_head_chest')
    ocvs = HeadChest()
    ocvs.run()
    rospy.spin() 

