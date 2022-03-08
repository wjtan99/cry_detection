from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import resampy

import soundfile as sf 
import pickle

from vad import vad 

#Follow Google Research VGG-ish AudioSet 
def generate_log_spectrogram(filepath,outpath,sr=16000,duration=5.0,segment=1.0,pre_emphasis=None,
                            n_mels=64,fmin=0,fmax=None,n_fft=256, hop_length=128, debug=False,VAD=False):
    
    save_as_filetype = 0  # 0 - not save, 1-png, 2-npy  
    
    fn = filepath.split('/')[-1] 
    step = np.ceil(segment)
    
    if duration == None:
        data,sampling_rate = librosa.load(filepath,sr=sr)     
        duration = librosa.get_duration(y=data, sr=sr)    
        
    num_imgs = int(duration/step)
    melgrams = [] 
    
    if debug: 
        print(filepath,fn)
        print(duration,segment,step,num_imgs) 
   
    
    for i in range(num_imgs):

        if save_as_filetype ==1:
            img_filepath = outpath + fn[:-3]+'_{}.npy'.format(i+1)
            if os.path.exists(img_filepath):
                continue 
        
        elif save_as_filetype ==2:
            img_filepath = outpath + fn[:-3]+'_{}.png'.format(i+1)
            if os.path.exists(img_filepath):
                continue        
        
        
        try: 
            data, sampling_rate = librosa.load(filepath,sr=sr,offset=i*step,duration=segment)               
            if not pre_emphasis is None:
                data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
            
            vocal = True 
            if VAD: 
                vocal = vad(data,sr=sampling_rate,segment=segment)[0]

            melgram = [] 
            
            if VAD==False or (VAD==True and vocal==True) or True:  
                melspectrogram = librosa.feature.melspectrogram(y=data, sr=sampling_rate,n_mels=n_mels, 
                                                            fmin=fmin,fmax=fmax, 
                                                            center=False,  
                                                            n_fft=n_fft, hop_length=hop_length)
                melgram = librosa.power_to_db(melspectrogram, ref=np.max)
                
                if debug: 
                    print(i,i*step,vocal,melgram.shape) 
                
                mmin = np.min(melgram)
                mmax = np.max(melgram)
                
                if mmax==mmin:
                    print(filepath,fn)
                    print(melgram)
                    #input("dbg melgram")                
                    continue 
                
                melgram = (melgram-mmin)/(mmax-mmin)*255
                
            melgrams.append((melgram,i,vocal))
            
            
            if save_as_filetype==1: 
                #print(melspectrogram.shape) 
                plt.figure(figsize=(1, 1))
                plt.figure()
                plt.axis('off')
                #print(melgram.shape,np.max(melgram),np.min(melgram),np.mean(melgram))
                librosa.display.specshow(melgram)                
                plt.savefig(img_filepath) #, dpi=224)
                plt.close()
            elif save_as_filetype==2: 
                np.save(img_filepath,melgram)
                #print(img_filepath)
            
        except:
            print("Reading audio file {} has error".format(filepath))

      
    return melgrams




if __name__ == '__main__':

    melgram = generate_log_spectrogram('./1-22694-A.ogg','./',sr=8000,segment=4.1, pre_emphasis=None,
                              n_mels=64,fmin=0,fmax=None,                                     
                              n_fft=512, hop_length=512,debug=True,VAD=True)
    print(melgram) 










