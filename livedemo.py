#!/usr/bin/env python
# coding: utf-8

# ## Record Audio

# In[1]:

import tensorflow as tf
import wave
from tensorflow import keras
import pyaudio
import pandas as pd
import numpy as np
import time

import IPython.display as ipd

while True:
   
# In[2]:


    CHUNK = 1024 
    FORMAT = pyaudio.paInt16 #paInt8
    CHANNELS = 2 
    RATE = 44100 #sample rate
    RECORD_SECONDS = 4
    WAVE_OUTPUT_FILENAME = "demo_audio.wav"
    emotions=["Anger","disgust","fear","happy","Neutral", "sad", "surprise"]
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK) #buffer
    
    print("* recording")
    
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data) # 2 bytes(16 bits) per channel
    
    print("* done recording")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    
    # ## Load The SER Model
    
    # In[3]:
    
    
    # loading json and creating model
    from keras.models import model_from_json
    json_file = open('./utils/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("./Trained_Models/Speech_Emotion_Recognition_Model.h5")
    print("Loaded model from disk")
    
    
    # In[4]:
    
    
    demo_audio_path = r'K:\thesis\speech-emotion-recognition-master\Codes\demo_audio.wav'
    ipd.Audio('demo_audio.wav')
    
    
    # In[5]:
    
    
    from utils.feature_extraction import get_audio_features
    demo_mfcc, demo_pitch, demo_mag, demo_chrom = get_audio_features(demo_audio_path,20000)
    
    
    # In[6]:
    
    
    mfcc = pd.Series(demo_mfcc)
    pit = pd.Series(demo_pitch)
    mag = pd.Series(demo_mag)
    C = pd.Series(demo_chrom)
    demo_audio_features = pd.concat([mfcc,pit,mag,C],ignore_index=True)
    
    
    # In[7]:
    
    
    demo_audio_features= np.expand_dims(demo_audio_features, axis=0)
    demo_audio_features= np.expand_dims(demo_audio_features, axis=2)
    
    
    # In[8]:
    
    
    demo_audio_features.shape
    
    
    # In[9]:
    
    
    livepreds = loaded_model.predict(demo_audio_features, 
                             batch_size=32, 
                             verbose=1)
    
    
    # In[10]:
    
    
    livepreds
    
    
    
    # In[17]:
    
    
    print(livepreds[0])
    
    
    # In[24]:
    
    
    emotions=["anger","disgust","fear","happy","neutral", "sad", "surprise"]
    index = livepreds.argmax(axis=1).item()
    index
    
    
    # In[25]:
    
    
    print(emotions[index])
   
 
    time.sleep(3)

# In[ ]:




