#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time, os, sys, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
import numpy as np


# In[2]:


from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip

from pathlib import Path
import librosa, librosa.display
import resampy

import soundfile as sf 
import pickle 


# In[6]:


import torch
import torch.utils.data as data
from torchvision import transforms

from model import BlazeNet 
from dataset import AudioDataset
from spectrogram import generate_log_spectrogram
from video_utils import * 


# In[7]:


output_dir = 'output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# In[8]:


test_transform = transforms.Compose([
    #transforms.Resize(size=128),
    #transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# In[9]:


# Instantiate a neural network model 
model_ft = BlazeNet(back_model=2)
model_ft = torch.load("checkpoints/blazenet_trainset_8000_4.1_None_64_0_None_512_512.pk.pth")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = model_ft.to(device)
model_ft.eval()


# In[10]:


sr=8000; segment=4.1; pre_emphasis=None
n_mels=64; fmin=0; fmax=None
n_fft=512; hop_length=512
VAD_on = True 


# In[20]:


def test_cry_detection_on_video(video_file,sr=8000,segment=4.1,pre_emphasis=None,
                                n_mels=64,fmin=0,fmax=None,n_fft=512,hop_length=512,
                                VAD_on=True,verbose=False):
    
    video_file_annot = video_file +'_crydetected.MP4'
    if os.path.exists(video_file_annot):
        return 0 
    
    
    print("Step 1.1: Extracting audio file from video file {}".format(video_file))
    audio_file = extract_audio(video_file) 
    
    print("Step 1.2: Read audio file {} and resample to {}".format(audio_file,sr))     
    audio_data, sr = librosa.load(audio_file,sr=sr)
    
    if verbose:        
        duration = librosa.get_duration(y=wav_data, sr=sr)    
        print("sampling rate = {}, length = {}, durations ={}s".format(sr,len(audio_data),duration))        
        plt.figure(1)
        plt.title("Signal Wave...")
        plt.plot(audio_data)
        plt.show()  
        
    print("Step 2.1: Generating log-mel-spectrogram")
    melgrams = generate_log_spectrogram(audio_file,None,sr=sr,duration=None,segment=segment,pre_emphasis=pre_emphasis,
                                    n_mels=n_mels,fmin=fmin,fmax=fmax,                                     
                                    n_fft=n_fft, hop_length=hop_length,VAD=VAD_on,debug=True)
    
    if len(melgrams)>0 and verbose: 
        print(len(melgrams))
        for m in melgrams:
            if m[2]:
                print(m[0].shape,m[1],m[2])
                break
                    

    print("Step 2.2 Saving into test dataset")
    test_data = {"test": {"cry":[],"nocry":[]}
                }
    test_data["test"]["cry"].append((audio_file,melgrams))
    dataset_file = output_dir+'testset_{}_{}_{}_{}_{}_{}_{}_{}_{}.pkl'.format(audio_file.split('/')[-1],
                                                                             sr,segment,pre_emphasis,n_mels,
                                                                             fmin,fmax,n_fft,hop_length)     
    
    print(dataset_file)
    with open(dataset_file , 'wb') as pk_file:
        pickle.dump(test_data, pk_file)        
    
    test_dataset2 = AudioDataset(dataset_file,
                                 subset="test",
                                 mode = "RGB",
                                 transform = test_transform)
    test_loader2 = data.DataLoader(test_dataset2,
                                   batch_size=32,
                                   shuffle=False,
                                   num_workers=4)

    if verbose: 
        for i, (img,label,src,ind,vocal) in enumerate(test_loader2):
            print(i)
            print(img.shape)
            print(label)
            print(src)
            print(ind)
            print(vocal)
            break  
            
    if len(test_dataset2)==0:
        print("Found no cry in video ",video_file)
        return 1 
    
    print("Step 3: Running cry detection CNN model")
    predictions = [] 
    with torch.no_grad():
        for images, labels, srcs, inds, vocals in test_loader2:
            images, labels = images.to(device), labels.to(device)
            outputs = model_ft(images)
            _, predicted = torch.max(outputs.data, 1)
            probs = torch.softmax(outputs.data, 1)[:,0]
            
            probs = probs.cpu().detach().numpy()
            inds = inds.detach().numpy()
            vocals = vocals.detach().numpy()

            if verbose:
                print(srcs)
                print(probs)
                print(inds)
                print(vocals)        
                #print(predicted)

            pred = predicted.tolist()       
            
            for k in range(len(srcs)):
                predictions.append((srcs[k],probs[k],inds[k],vocals[k])) # 1 is cry 
                

    predictions_by_audio = {} 
    cry_thresh = 0.5 
    audio_files = [x[0] for x in predictions]
    audio_files = list(set(audio_files))
    
    for au in audio_files:
        predictions_by_audio[au] = [] 
        
    for p in predictions: 
        predictions_by_audio[p[0]].append((p[1],p[2],p[3],p[1]>cry_thresh))
    for au in predictions_by_audio:
        print(au)
        print(predictions_by_audio[au])
        pred = predictions_by_audio[au]

    print("Step 4: Annotating the video file")    
    subs = [] 
    step = np.ceil(segment)
    for i in range(len(pred)):    
        s = pred[i][1]*step
        t = (pred[i][1]+1)*step
        if pred[i][3]:
            subs.append(((s,t),'Cry'))
        #else:
        #    subs.append(((s,t),'nocry'))
    print(subs)
    generator = lambda txt: TextClip(txt, font='Arial', fontsize=48, color='red')
    subtitles = SubtitlesClip(subs, generator)
    
    video = VideoFileClip(video_file)
    result = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])    
    result.write_videofile(video_file_annot, fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")
    
    return 2 


# In[19]:


#video_file = 'Self/10Cry/E062904F3842_subjectawake_1637717631593.mp4'
video_dir = 'Self/10Cry/'
video_files = os.listdir(video_dir)
video_files = [x for x in video_files if (x.endswith('mp4') or x.endswith('MP4')) and not 'crydetected' in x]
print(video_files)


# In[21]:


for v in video_files: 
    video_file = video_dir+v 
    test_cry_detection_on_video(video_file,sr=sr,segment=segment,pre_emphasis=pre_emphasis,
                                n_mels=n_mels,fmin=fmin,fmax=fmax,n_fft=n_fft,hop_length=hop_length,
                                VAD_on = VAD_on)
    


# In[ ]:




