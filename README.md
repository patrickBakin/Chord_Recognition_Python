# Chord_Recognition_Python

Predicting some piano Triad Chords(Major,Minor,Augmented,Diminished) from input source using Convolutional Networks 

This project does a rough recognition/classification of an input chord using deep learning, which requires users to input 1 chord sound file .wav before data preparation step.

## Brief Explanation on How it works

The Preparation step is to create folders with 48 different chords each folder contains their chord sounds then we split all files in each folder by 80:20 training:validation in order to apply augmentation to training data, what we're adding to the sounds are some noise and time-strecthing and offset shifting
then we load the augmented training data into numpy array and save it as compressed numpy format. (before saving each chords are labeled accords to its chord name)

The training step, all data are loaded into numpy array then we reshape the input arrays so it's compatible with networks layers, and we do one-hot encoding to outputs data corresponds to inputs, then we create a model stucture you can try adjusting the values or manipulating layers make it perform well on your data if needed. for me, I'm using this dataset [Audio Piano Triads Dataset](https://zenodo.org/record/4740877#.Yn-lluhByHt) Finally, the last part is training, I've add a Reduce Learning Rate on Plateu just for preventing gradient overshooting

## How to Run
1.Download Dataset from the link mentioned above, create a folder named "Data" in the same directory as this project then put "audio_augmented_x10" (a folder that u extract from the file you downloaded) in side "Data"

2.Create a folder named "audio_augmented_x10_val" next to "audio_augmented_x10"

3.Run Data_Pre-Processing.py --> Training.py

## Note
If you use your own dataset, you have to change some parameters or variables to match with your data shape
