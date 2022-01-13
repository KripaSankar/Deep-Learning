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
from keras.layers import Input,  LeakyReLU, Flatten, Dense, Dropout, Lambda, Conv2D, ZeroPadding2D, MaxPooling2D, concatenate, Activation, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, MaxPooling2D, concatenate, UpSampling2D, Reshape, merge, AveragePooling2D, Activation
from keras.layers import  ZeroPadding2D,  BatchNormalization
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
path = "./ksankara/ALL_LAYERS/models/generated_dcgan/fmnist/"


#######################################################################################################################################
## SETUP DCGAN HELPER FUNCTIONS
#######################################################################################################################################

class ImageHelper(object):
    def save_image(self, generated, epoch, directory):
        fig, axs = plt.subplots(5, 5)
        count = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(generated[count, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                count += 1
        fig.savefig("{}/{}.png".format(directory, epoch))
        plt.close()
        
    def makegif(self, directory):
        filenames = np.sort(os.listdir(directory))
        filenames = [ fnm for fnm in filenames if ".png" in fnm]
    
        with imageio.get_writer(directory + '/image.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(directory + filename)
                writer.append_data(image)
                
######################################################################################################################################
## SETUP DCGAN FUNCTIONS
#######################################################################################################################################


class DCGAN():
    def __init__(self, image_shape, generator_input_dim, image_helper, img_channels):
        optimizer = Adam(0.0002, 0.5)
        
        self._image_helper = image_helper
        self.img_shape = image_shape
        self.generator_input_dim = generator_input_dim
        self.channels = img_channels

        # Build models
        self._build_generator_model()
        self._build_and_compile_discriminator_model(optimizer)
        self._build_and_compile_gan(optimizer)

    def train(self, epochs, train_data, batch_size):
        
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        history = []
        for epoch in range(epochs):
            #  Train Discriminator
            batch_indexes = np.random.randint(0, train_data.shape[0], batch_size)
            batch = train_data[batch_indexes]
            genenerated = self._predict_noise(batch_size)
            loss_real = self.discriminator_model.train_on_batch(batch, real)
            loss_fake = self.discriminator_model.train_on_batch(genenerated, fake)
            discriminator_loss = 0.5 * np.add(loss_real, loss_fake)

            #  Train Generator
            noise = np.random.normal(0, 1, (batch_size, self.generator_input_dim))
            generator_loss = self.gan.train_on_batch(noise, real)

            # Plot the progress
            print ("---------------------------------------------------------")
            print ("******************Epoch {}***************************".format(epoch))
            print ("Discriminator loss: {}".format(discriminator_loss[0]))
            print ("Generator loss: {}".format(generator_loss))
            print ("---------------------------------------------------------")
            
            history.append({"D":discriminator_loss[0],"G":generator_loss})
            
            # Save images from every hundredth epoch generated images
            if epoch % 500 == 0:
              self._save_images(epoch)
                
        self._plot_loss(history)
        self._image_helper.makegif("./ksankara/ALL_LAYERS/models/generated_dcgan/fmnist/")        
    
    def _build_generator_model(self):
        generator_input = Input(shape=(self.generator_input_dim,))
        generator_seqence = Sequential(
                [Dense(128 * 7 * 7, activation="relu", input_dim=self.generator_input_dim),
                 Reshape((7, 7, 128)),
                 UpSampling2D(),
                 Conv2D(128, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 UpSampling2D(),
                 Conv2D(64, kernel_size=3, padding="same"),
                 BatchNormalization(momentum=0.8),
                 Activation("relu"),
                 Conv2D(self.channels, kernel_size=3, padding="same"),
                 Activation("tanh")])
    
        generator_output_tensor = generator_seqence(generator_input)       
        self.generator_model = Model(generator_input, generator_output_tensor)
        
    def _build_and_compile_discriminator_model(self, optimizer):
        discriminator_input = Input(shape=self.img_shape)
        discriminator_sequence = Sequential(
                [Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(64, kernel_size=3, strides=2, padding="same"),
                 ZeroPadding2D(padding=((0,1),(0,1))),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(128, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Conv2D(256, kernel_size=3, strides=2, padding="same"),
                 BatchNormalization(momentum=0.8),
                 LeakyReLU(alpha=0.2),
                 Dropout(0.25),
                 Flatten(),
                 Dense(1, activation='sigmoid')])
    
        discriminator_tensor = discriminator_sequence(discriminator_input)
        self.discriminator_model = Model(discriminator_input, discriminator_tensor)
        self.discriminator_model.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator_model.trainable = False
    
    def _build_and_compile_gan(self, optimizer):
        real_input = Input(shape=(self.generator_input_dim,))
        generator_output = self.generator_model(real_input)
        discriminator_output = self.discriminator_model(generator_output)        
        
        self.gan = Model(real_input, discriminator_output)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    def _save_images(self, epoch):
        generated = self._predict_noise(25)
        generated = 0.5 * generated + 0.5
        self._image_helper.save_image(generated, epoch, "./ksankara/ALL_LAYERS/models/generated_dcgan/fmnist/")
    
    def _predict_noise(self, size):
        noise = np.random.normal(0, 1, (size, self.generator_input_dim))
        return self.generator_model.predict(noise)
        
    def _plot_loss(self, history):
        hist = pd.DataFrame(history)
        plt.figure(figsize=(30,20))
        for colnm in hist.columns:
            plt.plot(hist[colnm],label=colnm)
        plt.legend()
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.show()
        
        
#######################################################################################################################################
# RUN MODEL
#######################################################################################################################################
##MNIST
#######

(X, _), (_, _) = fashion_mnist.load_data()
X_train = X / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

image_helper = ImageHelper()
generative_adversarial_network = DCGAN(X_train[0].shape, 100, image_helper, 1)
generative_adversarial_network.train(1000, X_train, batch_size=32)


#######################################################################################################################################
