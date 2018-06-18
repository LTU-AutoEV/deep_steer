#!/usr/bin/env python

###############
# ROS Imports #
###############

# Python libs
import sys, time, os

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError


##############
# DL Imports #
##############

import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from keras.utils import np_utils

import numpy as np
import math

VERBOSE=False
DISPLAY_IMG=True
CAM_SUB='/cam_pub/image_raw/compressed'

ROS_PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def radians_to_degrees(x):
    deg = math.degrees(x + math.pi)
    return deg

def degrees_to_radians(x):
    rad = math.radians(x) - math.pi
    return rad

class DeepSteer(object):

    def __init__(self, weights_path, hyperparameters, shape):
        self.weights_path = weights_path
        print('Loading model...')
        self.model = self._initModel(hyperparameters, shape)
        print('Loading weights...')
        self.model.load_weights(weights_path)
        print('Done!')

    def getAngleForImage(self, img):
        pred = self.model.predict(img)
        print('pred:')
        for p in pred:
            print(p)
        return pred[0][0] / 1000.0

    def _initModel(self, hyperparameters, shape):

        print(shape)
        base_model = InceptionV3(weights=None, include_top=False, input_shape=shape)

        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        # add a fully-connected layer
        x = Dense(hyperparameters["fc_size"], activation=hyperparameters["fc_activation"])(x)

        # add a logistic layer
        predictions = Dense(1, kernel_initializer='normal')(x)

        # train this model
        model = Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='adam', loss=hyperparameters["loss"], metrics=hyperparameters["metrics"])

        # fix model loading bug
        # https://github.com/keras-team/keras/issues/2397
        model._make_predict_function()

        print(model.summary())

        return model


class ROSImageSub:

    def __init__(self):
        '''Initialize ros subscriber'''
        # subscribed Topic
        self.subscriber = rospy.Subscriber(CAM_SUB,
            CompressedImage, self.callback,  queue_size = 1, buff_size=52428800)
        if VERBOSE :
            print "subscribed to %s" % CAM_SUB


    def callback(self, ros_data):
        '''Callback function of subscribed topic.'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format


        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        image_np = cv2.resize(image_np, (320,240))

        if DISPLAY_IMG:
            cv2.imshow('cv_img', image_np)
            cv2.waitKey(2)
            
        # Convert to single item tensor (3,32,32) => (1,3,32,32)
        image_np = np.expand_dims(image_np, axis=0)
        print('Input image shape', image_np.shape)

        turn = deepSteer.getAngleForImage(image_np)

        print(turn)

if __name__ == '__main__':

    # Hyperparameters
    hyperparameters = {
        "batchsize" : 32,
        "fc_size" : 1024,
        "fc_activation" : 'relu',
        "epoch_finetune" : 30,
        "epoch_transfer" : 30,
        "loss" : "mean_squared_error",
        "metrics" : None,#["accuracy"]
        "monitor" : 'val_loss'
    }

    weights_file = os.path.join(ROS_PKG_PATH, 'weights/final_weights.h5')

    # Create model
    global deepSteer
    deepSteer = DeepSteer(weights_file, hyperparameters, (240, 320, 3))
    # deepSteer = DeepSteer(weights_file, hyperparameters, (480, 640, 3))

    # Init ROS
    im_sub = ROSImageSub()
    rospy.init_node('deep_steer', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down ROS deep_drive')

    if DISPLAY_IMG: cv2.destroyAllWindows()
