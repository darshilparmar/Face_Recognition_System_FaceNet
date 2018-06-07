import os
import glob
import numpy as np 
import cv2 
import tensorflow as tf 
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K 

K.set_image_data_format('channels_first')


def triplet_loss(y_true,y_pred,alpha=0.3):
	anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis=-1)

	basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

	return loss 




def main():

	FRmodel = faceRecoModel(input_shape=(3,96,96))
	FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
	FRmodel.save('face-rec-Google.h5')
	print_summary(model)



main()