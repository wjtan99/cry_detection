from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import moviepy.editor as mp

from scipy.signal import butter, lfilter, freqz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


generator = lambda txt: TextClip(txt, font='Arial', fontsize=24, color='red')


def add_subtitle_to_video(pred,title,segment,video_file,video_file_output,verbose=False): 
    
    subs = []     
    for i in range(len(pred)):
        s = i*segment
        t = (i+1)*segment
        if pred[i]:
            subs.append(((s,t),title))
        else:
            subs.append(((s,t),"Non-"+title))
    if verbose:
        print(subs)
        
    subtitles = SubtitlesClip(subs, generator) 
    
    video = VideoFileClip(video_file)
    result = CompositeVideoClip([video, subtitles.set_pos(('center','bottom'))])
    result.write_videofile(video_file_output, fps=video.fps, remove_temp=True, codec="libx264", audio_codec="aac")    
    


def extract_audio(video_file,audio_dir=None,audio_file=None): 

    print("video_file = ",video_file) 

    video_file_splits = video_file.split('/') 
    video_dir = '/'.join(video_file_splits[:-1]) 
    video_filename = video_file_splits[-1] 


    if audio_dir is None:
        audio_dir = video_dir 

    if audio_file is None: 
        audio_filename = video_filename.replace('mp4','mp3')  

    audio_file = os.path.join(audio_dir,audio_filename) 
    #print(audio_file) 

    if not os.path.exists(audio_dir):
        os.makedirs(audio_dir)
    if os.path.exists(audio_file): #already exists 
        print(audio_file," already exists") 
        return audio_file     

    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)

    return audio_file  





