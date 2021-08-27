#!/usmesr/bin/env python
import argparse
import os

import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv_bridge import CvBridge, CvBridgeError


sys.path.append("/usr/lib/python2.7/dist-packages/")
import rospy
import rosbag

# from rospy import pcl
from sensor_msgs.msg import Image
import csv

cv_bridge = CvBridge()


def main():

    bag_file = "/home/vision/sy_ws/2021-08-18-11-12-21_chest.bag"
    out_dir = "/home/vision/sy_ws/"
    rgb_topic = "/chest/camera/color/image_raw"
    dep_topic = "/chest/camera/depth/image_rect_raw"
    # point_topic = "/chest/camera/depth/color/points"

    out_rgb_dir = os.path.join(out_dir, "rgb")
    out_dep_dir = os.path.join(out_dir, "depth")
    # out_point_dir = os.path.join(out_dir, "point")
    if not os.path.isdir(out_rgb_dir):
        os.makedirs(out_rgb_dir)
    if not os.path.isdir(out_dep_dir):
        os.makedirs(out_dep_dir)
    # if not os.path.isdir(out_point_dir):
    #     os.mkdir(out_point_dir)

    bag = rosbag.Bag(bag_file)

    topics = bag.get_type_and_topic_info()[1].keys()

    count = 0
    for topic, msg, t in bag.read_messages(topics=[rgb_topic, dep_topic]):
        fn = format(((rospy.rostime.Time.to_nsec(t) / 1e9) - 1580300000), ".6f")
        try:
            if topic == rgb_topic:
                cvimg = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                out_fn = os.path.join(out_rgb_dir, fn + ".png")

            elif topic == dep_topic:
                cvimg = cv_bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                # cvimg = cvimg.copy()*1000.0                # uncomment this only for astra
                # cvimg = cvimg.astype(np.uint16)            # uncomment this only for astra
                out_fn = os.path.join(out_dep_dir, fn + ".png")
                print("rgb image %i" % count)

            count += 1
            # cv2.imshow(topic, cvimg)
            cv2.imwrite(out_fn, cvimg)
            # cv2.waitKey(10)
        except CvBridgeError as e:
            print(e)
    # rgb_file.close()
    # dep_file.close()
    bag.close()


if __name__ == "__main__":
    # main()
    main()
