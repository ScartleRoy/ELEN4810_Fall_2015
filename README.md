# ELEN4810_Fall_2015
The course project for ELEN4810 in 2015 Fall semester, Columbia University.

===========================================================================
This project is about automatic singing transcription. The codes were written in Python.   

We use the librosa library$asd^1$ for generating the chromagram. We use the pYIN Vamp plugin$asdf2$ for pitch tracking. To successfully run the code, you need to first install librosa and Vamp. For librosa, you can install it by PyPI
```
pip install librosa
```
  
or   
```python
  git clone https://github.com/bmcfee/librosa.git
  easy_install librosa  
```

For Vamp, you may go through [this](https://code.soundsoftware.ac.uk/projects/vamp-plugin-sdk/wiki/Mtp1) website for the installation instructions. Currently our code only support MacOS, and please make sure the path for Vamp is   
```
  HOME/Library/Audio/Plug-Ins/Vamp/vamp-simple-host  
```
After installing Vamp, you need to install the pYIN plugin. It can be found [here](https://code.soundsoftware.ac.uk/projects/pyin). pYIN[1] is an algorithm that estimate the F0 of a monaural singing voice, and can estimate the vocal/non-vocal regions. Please follow the instructions on how to successfully install plugins for Vamp. You can find it on the website above.  


To run the demo for our code, simple do  
```python
  from transcribe_demo import *
  evaluate_path('wav') # evaluate all the .wav files in the wav folder
  evaluate_single('child1.wav') # evaluate a single .wav file  
```

You can use your own recordings to test the algorithm.  


----------------------------------------------------------------------------

[1] Mauch M, Dixon S. pYIN: A fundamental frequency estimator using probabilistic threshold distributions[C]//Acoustics, Speech and Signal Processing (ICASSP), 2014 IEEE International Conference on. IEEE, 2014: 659-663.

$1$https://github.com/bmcfee/librosa
$2$http://www.vamp-plugins.org/
