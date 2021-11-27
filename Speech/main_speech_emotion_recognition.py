#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/drive/1sfa3fN9W0nExpGwIwaX36arm1xYkH2PN" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Project at Glance
# 
# **1. Basic Setup** <br>
# **2. Installation of Dependencies**<br>
# **3. Data Preparation**<br>
# **4. Data Visualization**<br>
# **5. Data Pre-Processing**<br>
# **6. Model Creation**<br>
# **7. Training and Evaluation**<br>
# **8. Test Set Prediction**<br>
# **9. Live Demonstration** <br>
# 

# # About Project 

# *  first step is to actually load the data into a machine understandable format.
# * For this, we simply take values after every specific time steps. For example; in a 2 second audio file, we extract values at half a second. This is called sampling of audio data, and the rate at which it is sampled is called the sampling rate. 
# 
# ![SpeechWave](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/23210623/sound.png)
# 
# * Another way of representing audio data is by converting it into a different domain of data representation, namely the **frequency domain**
# 
# * sampling of data required much mire data points 
# * if we represent audio data in frequency domain, much less computational space is required. 
# 

# 
# * Now the next step is to extract features from this audio representations, so that our algorithm can work on these features and perform the task it is designed for. Here’s a visual representation of the categories of audio features that can be extracted.
# 
# ![Block Digram](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/23233229/audio-features.png)
# 
# * After extracting these features, it is then sent to the machine learning model for further analysis.
# 

# # 1. Basic Setup

# In[1]:


# Load the Drive helper and mount
#from google.colab import drive

# This will prompt for authorization.
#drive.mount('/content/drive')


# ### Adjusting Path for this Notebook

# In[2]:


#cd drive/'My Drive'/'Colab Notebooks'/'Emotion Speech Recognition'/


# In[3]:


# Now above cell output is our path which is in current working directory
get_ipython().system('ls')


# # 2. Installation of Dependencies
# 

# 
# Essential requirement of of our project :
# 1. **Python 3.7**
# 2. **Librosa **
# 3. **PyTorch **
# 4. **Keras**
# 5. **GPU**
# 
# We have Already installed this frameworks and packages.

# In[9]:


# Provides a way of using operating system dependent functionality. 
import os

# LibROSA provides the audio analysis
import librosa
# Need to implictly import from librosa
import librosa.display

# Import the audio playback widget
import IPython.display as ipd
from IPython.display import Image

# Enable plot in the notebook
#pylab inline
# matplotlib inline
import matplotlib.pyplot as plt

# These are generally useful to have around
import numpy as np
import pandas as pd


# To build Neural Network and Create desired Model
import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D #, AveragePooling1D
from keras.layers import Flatten, Dropout, Activation # Input, 
from keras.layers import Dense #, Embedding
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


# # 3. Data Preparation

# 
# ### Plotting the audio file's waveform and its spectrogram

# In[10]:


data, sampling_rate = librosa.load('Dataset/happy/happy012.wav')
# To play audio this in the jupyter notebook
ipd.Audio('Dataset/happy/happy012.wav')


# In[11]:


plt.figure(figsize=(15, 5))
librosa.display.waveplot(data, sr=sampling_rate)


# ### Setup the Basic Paramter

# In[12]:


dataset_path = os.path.abspath('./Dataset')
destination_path = os.path.abspath('./')
# To shuffle the dataset instances/records
randomize = True
# for spliting dataset into training and testing dataset
split = 0.8
# Number of sample per second e.g. 16KHz
sampling_rate = 20000 
emotions=["Angry","Happy","Neutral", "Sad", "Surprise"]


# ### Converting Dataset in CSV format
# 
# it will cause easy operation on Dataset.

# In[13]:


# loading dataframes using dataset module 
from utils import dataset
df, train_df, test_df = dataset.create_and_load_meta_csv_df(dataset_path, destination_path, randomize, split)


# In[14]:


print('Dataset samples  : ', len(df),"\nTraining Samples : ", len(train_df),"\ntesting Samples  : ", len(test_df))


# # 4. Data Visualization

# Let's understand what is our dataset.

# In[15]:


df.head()


# In[16]:


print("Actual Audio : ", df['path'][0])
print("Labels       : ", df['label'][0])


# 
# ### Labels Assigned for emotions : 
# - 0 : Angry
# - 1 : Happy
# - 2 : Neutral 
# - 3 : Sad
# - 4 : Surprise
# 

# In[17]:


unique_labels = train_df.label.unique()
unique_labels.sort()
print("unique labels in Emtion dataset : ")
print(*unique_labels, sep=', ')
unique_labels_counts = train_df.label.value_counts(sort=False)
print("\n\nCount of unique labels in Emtion dataset : ")
print(*unique_labels_counts,sep=', ')


# In[18]:


# Histogram of the classes
plt.bar(unique_labels, unique_labels_counts,align = 'center', width=0.6, color = 'c')
plt.xlabel('Number of labels', fontsize=16)
plt.xticks(unique_labels)
plt.ylabel('Count of each labels', fontsize=16)
plt.title('Histogram of the Labels', fontsize=16)
plt.show()


# # 5. Data Pre-Processing

# ### Getting the features of audio files using librosa
# 
# Calculating MFCC, Pitch, magnitude, Chroma features.

# In[19]:


Image('./images/feature_plots.png')


# In[20]:


#from utils.feature_extraction import get_features_dataframe
#from utils.feature_extraction import get_audio_features

#trainfeatures, trainlabel = get_the_features(train_df, sampling_rate)
#testfeatures, testlabel = get_the_features(test_df, sampling_rate)

# I have ran above 2 lines and get the featured dataframe. 
# and store it into pickle file to use it for later purpose.
# it take too much time to generate features(around 30-40 minutes).

trainfeatures = pd.read_pickle('./features_dataframe/trainfeatures')
trainlabel = pd.read_pickle('./features_dataframe/trainlabel')
testfeatures = pd.read_pickle('./features_dataframe/testfeatures')
testlabel = pd.read_pickle('./features_dataframe/testlabel')


# In[21]:


trainfeatures.shape


# In[22]:


trainfeatures = trainfeatures.fillna(0)
testfeatures = testfeatures.fillna(0)


# In[23]:


# By using .ravel() : Converting 2D to 1D e.g. (512,1) -> (512,). To prevent DataConversionWarning

X_train = np.array(trainfeatures)
y_train = np.array(trainlabel).ravel()
X_test = np.array(testfeatures)
y_test = np.array(testlabel).ravel()


# In[24]:


y_train[:5]


# In[25]:


# One-Hot Encoding
lb = LabelEncoder()

y_train = np_utils.to_categorical(lb.fit_transform(y_train))
y_test = np_utils.to_categorical(lb.fit_transform(y_test))


# In[26]:


y_train[:5]


# ### Changing dimension for CNN model

# In[27]:


x_traincnn =np.expand_dims(X_train, axis=2)
x_testcnn= np.expand_dims(X_test, axis=2)


# In[28]:


x_traincnn.shape


# # 6. Model Creation

# In[29]:


model = Sequential()

model.add(Conv1D(256, 5,padding='same',
                 input_shape=(x_traincnn.shape[1],x_traincnn.shape[2])))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)


# In[30]:


model.summary()


# In[31]:


model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])


# # 7. Training and Evaluation

# ### Removed the whole training part for avoiding unnecessary long epochs list

# In[32]:


cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=400, validation_data=(x_testcnn, y_test))


# ### Loss Vs Iterations

# In[33]:


plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# ### Saving the model

# In[34]:


model_name = 'Speech_Emotion_Recognition_Model.h5'
save_dir = os.path.join(os.getcwd(), 'Trained_Models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)


# In[35]:


import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)


# ### Loading the model

# In[36]:


# loading json and creating model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
score = loaded_model.evaluate(x_testcnn, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# # 8. Test Set Prediction

# ### Predicting emotions on the test data

# In[56]:


preds = loaded_model.predict(x_testcnn, 
                         batch_size=32, 
                         verbose=1)


# In[57]:


preds


# In[58]:


preds1=preds.argmax(axis=1)


# In[59]:


preds1


# In[60]:


abc = preds1.astype(int).flatten()


# In[61]:


predictions = (lb.inverse_transform((abc)))


# In[62]:


preddf = pd.DataFrame({'predictedvalues': predictions})
preddf[:10]


# In[63]:


actual=y_test.argmax(axis=1)
abc123 = actual.astype(int).flatten()
actualvalues = (lb.inverse_transform((abc123)))


# In[64]:


actualdf = pd.DataFrame({'actualvalues': actualvalues})
actualdf[:10]


# In[65]:


finaldf = actualdf.join(preddf)


# ## Actual v/s Predicted emotions

# In[66]:


finaldf[170:180]


# In[67]:


finaldf.groupby('actualvalues').count()


# In[68]:


finaldf.groupby('predictedvalues').count()


# In[69]:


finaldf.to_csv('Predictions.csv', index=False)


# # 9. Live Demonstration

# #### The file 'output10.wav' in the next cell is the file that was recorded live using the code in AudioRecoreder notebook found in the repository

# In[102]:


demo_audio_path = 'anger008.wav'
ipd.Audio('anger008.wav')


# In[101]:


demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path,sampling_rate)

mfcc = pd.Series(demo_mfcc)
pit = pd.Series(demo_pitch)
mag = pd.Series(demo_mag)
C = pd.Series(demo_chrom)
demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)


# In[92]:


demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
demo_audio_features= np.expand_dims(demo_audio_features, axis=2)


# In[93]:


demo_audio_features.shape


# In[94]:


livepreds = loaded_model.predict(demo_audio_features, 
                         batch_size=32, 
                         verbose=1)


# In[95]:


livepreds


# In[96]:


# emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
index = livepreds.argmax(axis=1).item()
index


# In[97]:


emotions[index]


# # Thank You !
