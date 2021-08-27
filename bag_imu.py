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
from sensor_msgs.msg import Image
import csv

cv_bridge = CvBridge()


def main(args):
    rospy.init_node("Extract_IMU_data_from_bag_node", anonymous=True)
    t0 = rospy.get_time()

    bag_file = "/home/vision/sy_ws/2021-08-04-18-07-33.bag"
    out_dir = "/home/vision/sy_ws/imu"
    imu_topic = "/lsy/camera/imu"
    rgb_topic = "/lsy/camera/color/image_raw"
    dep_topic = "/lsy/camera/depth/image_rect_raw"
    point_topic = "/lsy/camera/depth/color/points"


    out_imu_dir = os.path.join(out_dir, "imu")


    if not os.path.isdir(out_imu_dir):
        os.mkdir(out_imu_dir)

    bag = rosbag.Bag(bag_file)
    N = bag.get_message_count(imu_topic)  # number of measurement samples
    data = np.zeros((6, N))  # preallocate vector of measurements
    time_sample = np.zeros((2, N))  # preallocate vector of measurements

    cnt = 0
    avgSampleRate = 0
    for topic, msg, t in bag.read_messages(topics=[imu_topic]):
        data[0, cnt] = msg.linear_acceleration.x
        data[1, cnt] = msg.linear_acceleration.y
        data[2, cnt] = msg.linear_acceleration.z
        data[3, cnt] = msg.angular_velocity.x
        data[4, cnt] = msg.angular_velocity.y
        data[5, cnt] = msg.angular_velocity.z
        time_sample[0, cnt] = msg.header.stamp.secs * pow(10, 9) + msg.header.stamp.nsecs
        if cnt > 1:
            time_sample[1, cnt] = pow(10, 9) / (time_sample[0, cnt] - time_sample[0, cnt - 1])
            avgSampleRate = avgSampleRate + time_sample[1, cnt]
        cnt = cnt + 1

    sampleRate = avgSampleRate / (cnt - 3)
    bag.close()

    # print "[%0.2f seconds] Bagfile parsed\n"%(rospy.get_time()-t0)

    """""" """""" """
    " write to cvs"
    """ """""" """"""
    fname = "allan_A_xyz_G_xyz_"
    f = open(out_imu_dir + fname + "rate_%f.csv" % sampleRate, "wt")

    try:
        writer = csv.writer(f)
        writer.writerow(("Time", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"))

        for i in range(data[1, :].size):
            writer.writerow(
                (
                    time_sample[0, i],
                    data[0, i],
                    data[1, i],
                    data[2, i],
                    data[3, i],
                    data[4, i],
                    data[5, i],
                )
            )
    finally:
        f.close()


if __name__ == "__main__":
    main(sys.argv)
