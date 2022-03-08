import librosa, librosa.display
from matplotlib import pyplot as plt
import numpy as np

import moviepy.editor as mp

from video_utils import * 

def vad(audio_data,sr=8000,segment=5,thresh=[0.25,0.025],verbose=False):
    
    frame_len = int(sr*segment)
    #print(frame_len) 

    audio_len = len(audio_data) 
    data = np.abs(audio_data) 
    
    
    num_frames = int(audio_len/frame_len)
    energys = [] 
    noises = [] 
    peaks = [] 
    
    if num_frames<1:
        return [] 
    
    for i in range(num_frames):
        frame = data[i*frame_len:(i+1)*frame_len] 
        energy = np.mean(np.square(frame))
        noise  = np.var(frame)
        energys.append(energy)
        noises.append(noise)
        peaks.append(np.max(frame)) 
    
    max_energy = max(energys)
    delta = 2*min(max_energy/5,1e-5)

    noises2 = [x+delta for x in noises]
    snr =  [x[0]/x[1] for x in zip(energys,noises2)]


    if verbose: 
        plt.figure(11)
        plt.plot(energys,'r')
        plt.plot(noises,'b')
        plt.show()        
        print("max_energe = ",max_energy)
        plt.figure(12)
        plt.plot(snr)
        plt.show()  
        
    silence = [x[0]>thresh[0] and x[1]>thresh[1] for x in zip(snr,peaks)]

    return silence


#a wrapper to read a audio file 
def vad_on_audio_file(audio_data,sr=8000,segment=5,thresh=[0.25,0.025],verbose=False):
    
    segment = segment
    audio_data, sr = librosa.load(audio_file,sr=sr)
    if verbose: 
        duration = librosa.get_duration(y=wav_data, sr=sr)    
        print("sampling rate = {}, length = {}, durations ={}s".format(sr,len(wav_data),duration))

    return vad(audio_data,sr=sr,segment=segment,thresh=segment,verbose=verbose)


def vad_on_video(video_file,sr=8000,segment=5,thresh=[0.25,0.1],output_dir=None,verbose=False): 

    video_file_splits = video_file.split('/') 
    video_dir = '/'.join(video_file_splits[:-1]) 
    video_filename = video_file_splits[-1] 

    if output_dir is None: 
        output_dir = video_dir     

    video_file_output = os.path.join(output_dir,video_filename+'_vad_annotated.mp4') 
    print(video_file_output) 

    if os.path.exists(video_file_output) and verbose is False: #already exists 
        return video_file_output     
 

    #extract audio from video file 
    audio_file = extract_audio(video_file,audio_dir=output_dir) 
    print(audio_file) 

    segment = segment
    wav_data, sr = librosa.load(audio_file,sr=sr)
    duration = librosa.get_duration(y=wav_data, sr=sr)    
    print("sampling rate = {}, length = {}, durations ={}s".format(sr,len(wav_data),duration))

    if verbose: 

        plt.figure(1)
        plt.title("Signal Wave...")
        plt.plot(wav_data)
        plt.show()  

        pre_emphasis = 0.97 
        wav_data = np.append(wav_data[0], wav_data[1:] - pre_emphasis * wav_data[:-1])
        plt.figure(2)
        plt.title("Signal Wave...")
        plt.plot(wav_data)
        plt.show()  
  
    vocal = vad(wav_data, sr=sr,segment=segment,thresh=thresh,verbose=verbose)
    

    video_file_output = os.path.join(output_dir,video_filename+'_vad_annotated.mp4') 
    print(video_file_output) 

    add_subtitle_to_video(vocal,"Vocal",segment,video_file,video_file_output,verbose=verbose) 

    return video_file_output   


if __name__ == "__main__":

    sr=8000 
    segment=5 #seconds 
    thresh = [0.25,0.025] #these thresholds are set loose. It is okay to detect silence as vocal, but not the other way.  
    verbose = False 

    ''' 
    #test audio file   
    print("testing vad on audio file") 
    video_file = 'Self/10Cry/146B9CB78A18_subjectawake_1620132682993.mp4'
    audio_file = 'Self/10Cry/146B9CB78A18_subjectawake_1620132682993.mp3'
    video_file_output = 'Self/10Cry/146B9CB78A18_subjectawake_1620132682993_vocal_annotated.mp4'

    output_dir = 'output'

    print("reading audio file ... ") 
    audio_data, sr = librosa.load(audio_file,sr=8000)
    duration = librosa.get_duration(y=audio_data, sr=sr)    
    print("sampling rate = {}, length = {}, durations ={}s".format(sr,len(audio_data),duration))

    vocal = vad(audio_data,sr=sr,segment=segment,thresh=thresh,verbose=False)
    add_subtitle_to_video(vocal,"Vocal",segment,video_file,video_file_output,verbose=verbose) 

    input("Press any key to test vad on vidoe file") 
    '''
    

     
    video_dir = 'Self/10Cry/' 
    video_files = os.listdir(video_dir) 
    video_files = [x for x in video_files if x.endswith('mp4') ] 
    print(video_files) 

    output_dir = 'output' 
    for v in video_files:
        v_full = video_dir+v 
        vad_on_video(v_full,sr=sr,segment=segment,thresh=thresh,output_dir=output_dir,verbose=False) 

     
    ''' 
    video_file = 'Self/10Cry/E06290634162_subjectawake_1638677765467.mp4'
    vad_on_video(video_file,sr=sr,segment=segment,thresh=thresh,output_dir='output',verbose=True) 
    ''' 

