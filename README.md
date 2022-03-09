# Baby Cry Detection 

Baby cry detection detects baby cry in a video file, and add the results as subtitle and generate an output video.
It includes audio activity detection (VAD) and a CNN model on the log-mel-spectrogram.   

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependent packages 

```bash
pip install librosa 
pip install moviepy 
pip soundfile, resampy 
```
It also depends on ffmpeg, and imageMagicK. You need to modify a imageMagicK permission file.   


## Usage

1. Change the video source file directory 
2. Run python test-cry-detection-video.py 

