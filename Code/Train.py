#!/usr/bin/env python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import tensorflow as tf
import cv2
import sys
import os
import glob
# import Misc.ImageUtils as iu
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
from Network.Network import HomographyModel
from Misc.MiscUtils import *
from Misc.DataUtils import *
import numpy as np
import time
import argparse
import shutil
from StringIO import StringIO
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
from Misc.TFSpatialTransformer import *

# Don't generate pyc codes
sys.dont_write_bytecode = True


def imgCorners(img):
    # gray = np.float32(img)
    features = cv2.goodFeaturesToTrack(img, 1500, 0.02,10)
    # h,w = img.shape[0],img.shape[1]
    Nstrong = features.shape[0]
    if(Nstrong >= 5):
    	print("Found a match")
    	return True
    else:
    	print("Reject this match --- Found only "+str(Nstrong)+" matches")
    	return False

def GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize):
	"""
	Inputs: 
	BasePath - Path to COCO folder without "/" at the end
	DirNamesTrain - Variable with Subfolder paths to train files
	NOTE that Train can be replaced by Val/Test for generating batch corresponding to validation (held-out testing in this case)/testing
	TrainLabels - Labels corresponding to Train
	NOTE that TrainLabels can be replaced by Val/TestLabels for generating batch corresponding to validation (held-out testing in this case)/testing
	ImageSize - Size of the Image
	MiniBatchSize is the size of the MiniBatch
	Outputs:
	I1Batch - Batch of images
	LabelBatch - Batch of one-hot encoded labels 
	"""
	I1Batch = []
	LabelBatch = []
	img_size = 64
	perturb_size = img_size/4

	ImageNum = 0
	while ImageNum < MiniBatchSize:
		# Generate random image
		RandIdx = random.randint(0, len(DirNamesTrain)-1)
		
		RandImageName = BasePath + os.sep +'Data/'+ DirNamesTrain[RandIdx] + '.jpg'   
		ImageNum += 1
		
		##########################################################
		# Add any standardization or data augmentation here!
		##########################################################

		im = np.float32(cv2.imread(RandImageName,0))
		# Label = convertToOneHot(TrainLabels[RandIdx], 10)
		h,w = im.shape
		# im = cv2.imread(RandImageName,0)
		
		flag= False
		while(not flag):
			x_ , y_ = random.randint(h/4,h - img_size), random.randint(w/4, w - img_size)
			if((x_ + img_size)<=(h-perturb_size/2) and (y_ + img_size)<=(w-perturb_size/2)):
				while(not flag):
					patch = im[x_:x_ + img_size, y_:y_ + img_size]
					flag = True #imgCorners(patch)


		# x_ , y_ = random.randint(h/2,3*h/4), random.randint(w/2,3*w/4) 
		# patch = im[x_:x_ + img_size, y_:y_ + img_size]
		# u,v = []  ## U is --- x ---  and ---  V is Y ---
		u = [random.randint(-perturb_size/2,perturb_size/2) for i in range(4)]
		v = [random.randint(-perturb_size/2,perturb_size/2) for i in range(4)]
		# Append All Images and Mask
		
		im1 = im[x_:x_+img_size, y_:y_+img_size]
		pa = np.array([[y_,x_],[y_+img_size,x_],[y_+img_size,x_+img_size],[y_,x_+img_size]], dtype='f')
		pb = np.array([[y_+u[0],x_+v[0]],[y_+img_size+u[1],x_+v[1]],[y_+u[2]+img_size,x_+img_size+v[2]],[y_+u[3],x_+img_size+v[3]]], dtype='f')
		H = np.linalg.inv(cv2.getPerspectiveTransform(pa,pb))
		im2_ = cv2.warpPerspective(im,H,(w,h))
		im2 = im2_[x_:x_+img_size, y_:y_+img_size]
		im_in = np.zeros((img_size,img_size,2))
		
		im_in[:,:,0] = (im1 - im1.mean())/im1.std()
		im_in[:,:,1] = (im2 - im2.mean())/im2.std()
		
		output_homo = np.array(u + v)
		I1Batch.append(im_in)
		LabelBatch.append(output_homo)
	return I1Batch, LabelBatch



def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
	"""
	Prints all stats with all arguments
	"""
	print('Number of Epochs Training will run for ' + str(NumEpochs))
	print('Factor of reduction in training data is ' + str(DivTrain))
	print('Mini Batch Size ' + str(MiniBatchSize))
	print('Number of Training Images ' + str(NumTrainSamples))
	if LatestFile is not None:
		print('Loading latest checkpoint with the name ' + LatestFile)              

	
def TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType):
	"""
	Inputs: 
	ImgPH is the Input Image placeholder
	LabelPH is the one-hot encoded label placeholder
	DirNamesTrain - Variable with Subfolder paths to train files
	TrainLabels - Labels corresponding to Train/Test
	NumTrainSamples - length(Train)
	ImageSize - Size of the image
	NumEpochs - Number of passes through the Train data
	MiniBatchSize is the size of the MiniBatch
	SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
	CheckPointPath - Path to save checkpoints/model
	DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
	LatestFile - Latest checkpointfile to continue training
	BasePath - Path to COCO folder without "/" at the end
	LogsPath - Path to save Tensorboard Logs
	ModelType - Supervised or Unsupervised Model
	Outputs:
	Saves Trained network in CheckPointPath and Logs to LogsPath
	"""      
	# Predict output with forward pass
	prLogits, prSoftMax = HomographyModel(ImgPH, ImageSize, MiniBatchSize)

	with tf.name_scope('Loss'):
		###############################################
		# Fill your loss function of choice here!
		###############################################
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = LabelPH, logits = prLogits)
		# loss = tf.reduce_mean(cross_entropy)
		loss = tf.nn.l2_loss(cross_entropy)
		# loss = tf.reduce_mean(loss_)
		
	with tf.name_scope('Accuracy'):
		prSoftMaxDecoded = tf.argmax(prSoftMax, axis=1)
		LabelDecoded = tf.argmax(LabelPH, axis=1)
		Acc = tf.reduce_mean(tf.cast(tf.math.equal(prSoftMaxDecoded, LabelDecoded), dtype=tf.float32))

	with tf.name_scope('Adam'):
		###############################################
		# Fill your optimizer of choice here!
		###############################################
		Optimizer = tf.train.AdamOptimizer(learning_rate = 3*1e-3).minimize(loss)

	# Tensorboard
	# Create a summary to monitor loss tensor
	tf.summary.scalar('LossEveryIter', loss)
	tf.summary.scalar('Accuracy', Acc)
	# tf.summary.image('Anything you want', AnyImg)
	# Merge all summaries into a single operation
	MergedSummaryOP = tf.summary.merge_all()

	# Setup Saver
	Saver = tf.train.Saver()
	acc = []
	temp_acc = []
	temp_loss = []
	loss_ = []
	with tf.Session() as sess:       
		if LatestFile is not None:
			Saver.restore(sess, CheckPointPath + LatestFile + '.ckpt')
			# Extract only numbers from the name
			StartEpoch = int(''.join(c for c in LatestFile.split('a')[0] if c.isdigit()))
			print('Loaded latest checkpoint with the name ' + LatestFile + '....')
		else:
			sess.run(tf.global_variables_initializer())
			StartEpoch = 0
			print('New model initialized....')

		# Tensorboard
		Writer = tf.summary.FileWriter(LogsPath, graph=tf.get_default_graph())
			
		for Epochs in tqdm(range(StartEpoch, NumEpochs)):
			NumIterationsPerEpoch = int(NumTrainSamples/MiniBatchSize/DivTrain)
			for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
				I1Batch, LabelBatch = GenerateBatch(BasePath, DirNamesTrain, TrainLabels, ImageSize, MiniBatchSize)
				FeedDict = {ImgPH: I1Batch, LabelPH: LabelBatch}
				_, LossThisBatch, Summary = sess.run([Optimizer, loss, MergedSummaryOP], feed_dict=FeedDict)
				temp_loss.append(LossThisBatch)
				temp_acc.append(sess.run([Acc], feed_dict=FeedDict))
				# Save checkpoint every some SaveCheckPoint's iterations
				if PerEpochCounter % SaveCheckPoint == 0:
					# Save the Model learnt in this epoch
					SaveName =  CheckPointPath + str(Epochs) + 'a' + str(PerEpochCounter) + 'model.ckpt'
					Saver.save(sess,  save_path=SaveName)
					print('\n' + SaveName + ' Model Saved...')
					print("Loss of model : "+str(LossThisBatch))
					print("Accuracy of model : " + str(sess.run([Acc], feed_dict=FeedDict)))
				# Tensorboard
				Writer.add_summary(Summary, Epochs*NumIterationsPerEpoch + PerEpochCounter)
				# If you don't flush the tensorboard doesn't update until a lot of iterations!
				Writer.flush()

			# Save model every epoch
			SaveName = CheckPointPath + str(Epochs) + 'model.ckpt'
			Saver.save(sess, save_path=SaveName)
			loss_.append(np.array(temp_loss).sum())
			acc.append(np.array(temp_acc).mean())
			print('\n' + SaveName + ' Model Saved...')
			print("----------------After epoch------------")
			print("Total loss = "+str(np.array(temp_loss).sum()))
			print("Total accuracy = "+str(np.array(temp_acc).mean()))
			print("--------------------------------------------")
			temp_acc = []
			temp_loss = []

def main():
	"""
	Inputs: 
	None
	Outputs:
	Runs the Training and testing code based on the Flag
	"""
	# Parse Command Line arguments
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--BasePath', default='/media/saumil/New Volume/CMSC_733/Proj1/YourDirectoryID_p1/Phase2', help='Base path of images, Default:/media/nitin/Research/Homing/SpectralCompression/COCO')
	Parser.add_argument('--CheckPointPath', default='../Checkpoints/', help='Path to save Checkpoints, Default: ../Checkpoints/')
	Parser.add_argument('--ModelType', default='Unsup', help='Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup')
	Parser.add_argument('--NumEpochs', type=int, default=50, help='Number of Epochs to Train for, Default:50')
	Parser.add_argument('--DivTrain', type=int, default=1, help='Factor to reduce Train data by per epoch, Default:1')
	Parser.add_argument('--MiniBatchSize', type=int, default=32, help='Size of the MiniBatch to use, Default:1')
	Parser.add_argument('--LoadCheckPoint', type=int, default=0, help='Load Model from latest Checkpoint from CheckPointsPath?, Default:0')
	Parser.add_argument('--LogsPath', default='Logs/', help='Path to save Logs for Tensorboard, Default=Logs/')

	Args = Parser.parse_args()
	NumEpochs = Args.NumEpochs
	BasePath = Args.BasePath
	DivTrain = float(Args.DivTrain)
	MiniBatchSize = Args.MiniBatchSize
	LoadCheckPoint = Args.LoadCheckPoint
	CheckPointPath = Args.CheckPointPath
	LogsPath = Args.LogsPath
	ModelType = Args.ModelType

	# Setup all needed parameters including file reading
	DirNamesTrain, SaveCheckPoint, ImageSize, NumTrainSamples, TrainLabels, NumClasses = SetupAll(BasePath, CheckPointPath)



	# Find Latest Checkpoint File
	if LoadCheckPoint==1:
		LatestFile = FindLatestModel(CheckPointPath)
	else:
		LatestFile = None
	
	# Pretty print stats
	PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

	# Define PlaceHolder variables for Input and Predicted output
	ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
	LabelPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, NumClasses)) # OneHOT labels
	
	TrainOperation(ImgPH, LabelPH, DirNamesTrain, TrainLabels, NumTrainSamples, ImageSize,
				   NumEpochs, MiniBatchSize, SaveCheckPoint, CheckPointPath,
				   DivTrain, LatestFile, BasePath, LogsPath, ModelType)
		
	
if __name__ == '__main__':
	main()
 
