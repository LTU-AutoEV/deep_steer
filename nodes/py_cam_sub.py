#!/usr/bin/env python

# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError

VERBOSE=False
DISPLAY_IMG=False
CAM_SUB='/cam_pub/image_raw/compressed'

class image_feature:

    def __init__(self):
        '''Initialize ros subscriber'''
        # subscribed Topic
        self.subscriber = rospy.Subscriber(CAM_SUB,
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print "subscribed to %s" % CAM_SUB


    def callback(self, ros_data):
        '''Callback function of subscribed topic.'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        print(image_np.shape)

        if DISPLAY_IMG:
            cv2.imshow('cv_img', image_np)
            cv2.waitKey(2)

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('image_feature', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
    if DISPLAY_IMG: cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
