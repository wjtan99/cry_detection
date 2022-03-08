#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import sys


# In[2]:


import random
random.seed(999)
random.uniform(0, 1)


# In[3]:


from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import resampy

import soundfile as sf 
import pickle 

from vad import *


# In[4]:


from spectrogram import generate_log_spectrogram


# In[5]:


audiofile = 'giulbia-data/301 - Crying baby/1-187207-A.ogg'
wav_data, sr = sf.read(audiofile)
print(wav_data[:10])
print(sr)
wav_data = resampy.resample(wav_data, sr, 16000)
print(wav_data[:10])

'''
wav_data1, sr1 = sf.read(audiofile,dtype='int16')
print(wav_data1[:10])
print(sr1)
wav_data1 = wav_data1/32768 
wav_data1 = resampy.resample(wav_data1, sr1, 16000)
print(wav_data1[:10])
'''

wav_data2, sr2 = librosa.load(audiofile,sr=16000)
print(wav_data2[:10])
print(sr2)


# In[6]:


generate_log_spectrogram(audiofile,'./',sr=8000,segment=4.1, pre_emphasis=None,
                                    n_mels=64,fmin=0,fmax=None,                                     
                                    n_fft=512, hop_length=512,debug=True,VAD=True)


# In[7]:


input("testing melgram") 


train_cry_sources = ['donateacry-corpus/deepcam-cleaned/crys-5s-segments/',
               'AudioSet/Babycry-infantcry-cleaned/babycry-5s-segments/'
              ] 

train_nocry_sources = [
                 'ESC-50/audio/',
                 'donateacry-corpus/deepcam-cleaned/filtered-out/',
                 'AudioSet/others_train_5s_segments/'
                ] 

valid_cry_sources = ['giulbia-data/301 - Crying baby/']
valid_nocry_sources = ['giulbia-data/901 - Silence/',
                       'giulbia-data/902 - Noise/',
                       'giulbia-data/903 - Baby laugh/',
                       'AudioSet/others_valid_5s_segments/'
                      ]
                     
test_cry_sources = ['ESC-50/babycry/',
                   'AudioSet/babycry_eval_5s_segments/'
                   ]

test_nocry_sources = ['AudioSet/others_eval_5s_segments/']


# In[8]:


'''
train_cry_sources = []

train_nocry_sources = [
                ] 

valid_cry_sources = ['giulbia-data/301 - Crying baby/']
valid_nocry_sources = ['giulbia-data/901 - Silence/',
                       'giulbia-data/902 - Noise/',
                       'giulbia-data/903 - Baby laugh/'
                       #'AudioSet/others_valid_5s_segments/'
                      ]
                     
test_cry_sources = ['ESC-50/babycry/']

test_nocry_sources = []
'''


# In[9]:


dataset = 'dataset' 

train_cry_folder = dataset + '/train/cry/'  
train_nocry_folder = dataset + '/train/nocry/'  

valid_cry_folder = dataset + '/valid/cry/'  
valid_nocry_folder = dataset + '/valid/nocry/'  

test_cry_folder = dataset + '/test/cry/'  
test_nocry_folder = dataset + '/test/nocry/'  

if not os.path.exists(train_cry_folder):
    os.makedirs(train_cry_folder)
if not os.path.exists(train_nocry_folder):
    os.makedirs(train_nocry_folder)
if not os.path.exists(valid_cry_folder):
    os.makedirs(valid_cry_folder)
if not os.path.exists(valid_nocry_folder):
    os.makedirs(valid_nocry_folder)
if not os.path.exists(test_cry_folder):
    os.makedirs(test_cry_folder)
if not os.path.exists(test_nocry_folder):
    os.makedirs(test_nocry_folder)

    


# In[10]:


cry_sources = [] 
cry_sources.append(train_cry_sources)
cry_sources.append(valid_cry_sources)
cry_sources.append(test_cry_sources)

noncry_sources = [] 
noncry_sources.append(train_nocry_sources)
noncry_sources.append(valid_nocry_sources)
noncry_sources.append(test_nocry_sources)


dataset = {"train": {"cry":[], "nocry":[] },
           "valid": {"cry":[], "nocry":[] },
           "test": {"cry":[], "nocry":[] }           
          }


# In[11]:


sr=8000; segment=4.1; pre_emphasis=None
n_mels=64; fmin=0; fmax=None
n_fft=512; hop_length=512
VAD_on = True 


# In[ ]:


for i in range(len(cry_sources)):
    source = cry_sources[i]    
    for s in source:
                
        files = os.listdir(s) 
        for f in files:
            
            r = random.uniform(0, 1)
            
            if r<0.8:
                dest = train_cry_folder
                dest_d = dataset["train"]["cry"]
            elif r<0.9:
                dest = valid_cry_folder
                dest_d = dataset["valid"]["cry"]
            else:
                dest = test_cry_folder
                dest_d = dataset["test"]["cry"]
                
            #print(s+f,dest)               
            melgrams = generate_log_spectrogram(s+f,dest,sr=sr,segment=segment,pre_emphasis=pre_emphasis,
                                    n_mels=n_mels,fmin=fmin,fmax=fmax,                                     
                                    n_fft=n_fft, hop_length=hop_length,VAD=VAD_on,debug=False)
            if len(melgrams)>0:
                dest_d.append((s+f,melgrams))


# In[14]:


for i in range(len(noncry_sources)):
    source = noncry_sources[i]    
    for s in source:
                
        files = os.listdir(s) 
        for f in files:
            
            r = random.uniform(0, 1)
            
            if r<0.8:
                dest = train_nocry_folder
                dest_d = dataset["train"]["nocry"]
            elif r<0.9:
                dest = valid_nocry_folder
                dest_d = dataset["valid"]["nocry"]
            else:
                dest = test_nocry_folder
                dest_d = dataset["test"]["nocry"]
                
            #print(s+f,dest)               
            melgrams = generate_log_spectrogram(s+f,dest,sr=sr,segment=segment,pre_emphasis=pre_emphasis,
                                    n_mels=n_mels,fmin=fmin,fmax=fmax,                                     
                                    n_fft=n_fft, hop_length=hop_length,VAD=VAD_on,debug=False)
            if len(melgrams)>0:
                dest_d.append((s+f,melgrams))


# In[22]:


dataset_file = 'dataset/trainset_{}_{}_{}_{}_{}_{}_{}_{}.pk'.format(
    sr,segment,pre_emphasis,n_mels,fmin,fmax,n_fft,hop_length) 

print(dataset_file)
with open(dataset_file , 'wb') as pk_file:
                pickle.dump(dataset, pk_file)


# In[16]:


test_dir = 'Self/10Cry/'
audiofiles = os.listdir(test_dir) 
audiofiles = [x for x in audiofiles if x.endswith('mp3')]
print(audiofiles)


# In[17]:


#audiofiles = ['E06290646B49_subjectAsleep_1638305984190.mp3']


# In[18]:


test_data = {"test":{"cry":[],"noncry":[]}
            }

for f in audiofiles:
    f_full = test_dir+f     
    print(f_full)
    melgrams = generate_log_spectrogram(f_full,None,sr=sr,duration=None,segment=segment,pre_emphasis=pre_emphasis,
                                    n_mels=n_mels,fmin=fmin,fmax=fmax,                                     
                                    n_fft=n_fft, hop_length=hop_length)
    
    if len(melgrams)>0:
        print(len(melgrams))
        test_data["test"]["cry"].append((f_full,melgrams))


# In[19]:


dataset_file = 'dataset/testset_{}_{}_{}_{}_{}_{}_{}_{}.pk'.format(
    sr,segment,pre_emphasis,n_mels,fmin,fmax,n_fft,hop_length) 

print(dataset_file)
with open(dataset_file , 'wb') as pk_file:
    pickle.dump(test_data, pk_file)        


# In[20]:


'''
dataset_file = 'dataset/testset_8000_5_None_64_0_None_256_128.pk'
with open(dataset_file, 'rb') as pk_file:
    data = pickle.load(pk_file)
    
for d in data:
    test_data["test"]["cry"].append((d[0],d[1]))
'''    

