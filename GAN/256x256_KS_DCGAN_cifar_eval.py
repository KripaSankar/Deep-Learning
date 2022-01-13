#######################################################################################################################################
## IMPORTS
#######################################################################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import keras
import pickle

from tabulate import tabulate
from IPython.display import display
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

from datetime import datetime
import warnings

from keras.models import Model, Sequential
from keras import regularizers
from keras.layers import Input,  LeakyReLU, Flatten, Dense, Dropout, Lambda, Conv2D, ZeroPadding2D, Conv2DTranspose, MaxPooling2D, concatenate, Activation, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Reshape, merge, AveragePooling2D, Activation
from keras.layers import  ZeroPadding2D,  BatchNormalization, Embedding, Concatenate
from keras.regularizers import l2
from keras.utils import multi_gpu_model
from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.layers import GaussianNoise
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, classification_report
from sklearn.utils.multiclass import unique_labels

from ksankara.utils.keras_data_generator import DataGenerator, DataGenerator_2Inputs
from ksankara.utils.misc import make_class_index_map_dict, create_input_shape
from ksankara.utils.keras_graph_helpers import plot_history, plot_confusion_matrix, plot_precision_recall, plot_history_hc, plot_history_hc_recall_by_class, plot_history_precision_recall_by_class
#from ksankara.utils.keras_graph_helpers import *
from ksankara.utils.misc import create_input_shape
from ksankara.utils.calculate_metrics import get_RF_POR_test_metrics, calculate_margin, get_margin_metrics, make_classification_report_dev, get_margin_metrics_v2, get_hc_marker
from ksankara.utils.calculate_metrics import  make_classification_report, HC_Metrics
from ksankara.utils.calculate_metrics import *
from ksankara.utils.global_objectives.keras_global_objectives import keras_recall_at_precision
from datetime import datetime
from scipy.special import softmax
#from imblearn import over_sampling as oversample
from imblearn.over_sampling import RandomOverSampler, SMOTE,ADASYN
from sklearn.utils import class_weight

from keras.datasets import fashion_mnist
from keras.datasets import cifar100
import imageio
from numpy import expand_dims
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import zeros
from numpy import ones
from keras.datasets.fashion_mnist import load_data
from keras.datasets.cifar10 import load_data

#######################################################################################################################################
## SET DISPLAY OPTIONS FOR PANDAS, MATPLOTLIB
#######################################################################################################################################

# set pandas dataframe display options
pd.options.display.html.table_schema = True
pd.options.display.max_columns = 500
pd.options.display.max_rows = 200

# set matplotlib defaults
matplotlib.rcParams['font.sans-serif'] = "FreeSans"
matplotlib.rcParams['font.family'] = "sans-serif"


#######################################################################################################################################
## SETUP PATH VARIABLES
#######################################################################################################################################
path = "./ksankara/ALL_LAYERS/models/generated_dcgan"


#######################################################################################################################################
## SETUP DCGAN 
#######################################################################################################################################

####################################
# DISCRIMINATOR 
####################################
# define the standalone discriminator model
def define_discriminator(in_shape=(32,32,3)):
	model = Sequential()
	# normal
	model.add(Conv2D(64, (3,3), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# downsample
	model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# classifier
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  #print("DISCRIMINATOR\n********")
  #model.summary()
	return model


####################################
#  GENERATOR
####################################
# define the standalone generator model
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 4x4 image
	n_nodes = 256 * 4 * 4
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((4, 4, 256)))
	# upsample to 8x8
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 16x16
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 32x32
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# output layer
	model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
  #print("GENERATOR\n********")
  #model.summary()
	return model
 
#####################################
#  DCGAN = DISCRIMINATOR + GENERATOR
######################################
# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model 

#######################################
# LOAD IMAGES
#######################################

#################
# REAL SAMPLES
#################
# Load the real samples
# load and prepare cifar10 training images

def load_real_samples():
	# load cifar10 dataset
	(trainX, _), (_, _) = load_data()
	# convert from unsigned ints to floats
	X = trainX.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X


# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

######################
# FAKE SAMPLES
######################

# Generator Input
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input


# Generate the samples 

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y
 
# create and save a plot of generated images
def save_plot(examples, epoch, n=7):
	# scale from [-1,1] to [0,1]
	examples = (examples + 1) / 2.0
	# plot images
	for i in range(n * n):
		# define subplot
		plt.subplot(n, n, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(examples[i])
	# save plot to file
	filename = './ksankara/ALL_LAYERS/models/generated_dcgan/cifar10_color/generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()
  


#######################
# EVALUATE PERFORMANCE
#######################

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=150):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch+1)
	g_model.save('./ksankara/ALL_LAYERS/models/generated_dcgan/dccgan_generator_cifar_color.h5')
  
  
######################
# TRAIN THE D AND G
######################

# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=250, n_batch=16):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# update discriminator model weights
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# update discriminator model weights
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
		# evaluate the model performance at regular intervals
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)


############################
# EXECUTE 
############################
# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
gan_model.summary()
#SVG(gan_model_to_dot(model,show_layer_names=True, show_shapes=True).create(prog='dot', format='svg'))
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

#############################
## LOAD MODEL and EXECUTE 
#############################
#
## load model
#model = load_model('./ksankara/ALL_LAYERS/models/generated_dcgan/dccgan_generator_cifar.h5')
## generate images
#latent_points, labels = generate_latent_points(100, 100)
## specify labels
#labels = asarray([x for _ in range(10) for x in range(10)])
## generate images
#X  = model.predict([latent_points, labels])
## scale from [-1,1] to [0,1]
#X = (X + 1) / 2.0
## plot the result
#save_plot(X, 10)
#
