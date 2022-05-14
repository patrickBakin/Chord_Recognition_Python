from keras.models import Sequential

from tensorflow import keras

import numpy as np

from keras.layers import Dense, Dropout
from keras.utils import np_utils

import sounddevice as sd
from keras.layers import Conv2D,Flatten,MaxPooling2D

import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile

import librosa.display
#from audiomentations import Compose,PitchShift,HighPassFilter,Trim


ChordDict = {'A Flat Aug': [], 'A Flat Dim': [], 'A Flat Maj': [], 'A Flat Min': [],
                 'A Aug': [], 'A Dim': [], 'A  Maj': [], 'A  Min': [],
                 'B Flat Aug': [], 'B Flat Dim': [], 'B Flat Maj': [], 'B Flat Min': [],
                 'B Aug': [], 'B Dim': [], 'B  Maj': [], 'B  Min': [],
                 'C Aug': [], 'C Dim': [], 'C Maj': [], 'C Min': [],
                 'D Flat Aug': [], 'D Flat Dim': [], 'D Flat Maj': [], 'D Flat Min': [],
                 'D Aug': [], 'D Dim': [], 'D  Maj': [], 'D  Min': [],
                 'E Flat Aug': [], 'E Flat Dim': [], 'E Flat Maj': [], 'E Flat Min': [],
                 'E Aug': [], 'E Dim': [], 'E Maj': [], 'E  Min': [],
                 'F Aug': [], 'F Dim': [], 'F Maj': [], 'F Min': [],
                 'G Flat Aug': [], 'G Flat Dim': [], 'G Flat Maj': [], 'G Flat Min': [],
                 'G Aug': [], 'G Dim': [], 'G  Maj': [], 'G  Min': []}



def Record():

    fs = 16000
    seconds = 2
    print("start record")
    myrecord = sd.rec(int(seconds * fs), samplerate=fs,channels=2)
    #print(myrecord)
    sd.wait()
    print("stop record")
    print(myrecord.shape)
    #plt.plot(myrecord[:, 0])
    #plt.plot(myrecord[:,1])
    #plt.show()
    sd.play(myrecord[:,0],samplerate=16000)
    wavfile.write("test_recording.wav",fs,myrecord[:,1])


def PrepareData(audiofile=None,sr=16000):
    
    new_frequency=Resampling(wave=audiofile,sr=sr)
    return new_frequency

def Resampling(wave=None,sr=0,audiodata=None):


    if audiodata is None:
        f, sr = librosa.load(wave, sr=None,offset=0.0, duration=0.9)
    else:
        f,sr=audiodata,sr

    output = librosa.resample(f, orig_sr=sr, target_sr=16000)
   
    st = librosa.stft(output, n_fft=1024, hop_length=256)

    
    spectogram = np.abs(st) ** 2
    spectogram = spectogram.round(decimals=5)

    # For plotting Spectogram
    """
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(spectogram, sr=16000, hop_length=256, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.f")
    plt.show()

    """

    return spectogram

def ProcessData(freqArray):
    New_freqArray=np.reshape(freqArray,(1,513,57,1))
    New_freqArray = (New_freqArray)/float(9957.391)
    output = np.array([])
    for i in range(48):
        a = np.array([i])
        b = np.repeat(a, repeats=720, axis=0)
        output = np.append(output, b)
    array_output = np_utils.to_categorical(output)

    return New_freqArray





def create_network():
        model = Sequential()
        model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(513,57,1)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dense(16))
        model.add(Dense(8))
        model.add(Flatten())
        model.add(Dense(48, activation="softmax"))
        adm = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])

        model.load_weights('weights-Chord-030-0.0052-0.9990.hdf5')

        return model


def DoPredict(model,prediction_input):

    prediction =model.predict(prediction_input)
    prediction_array_sorted = np.copy(prediction)
    prediction_array_sorted[0].sort()
    first_possible_chord=get_nth_key(ChordDict,n=np.where(prediction[0]==prediction_array_sorted[0][-1])[0][0])
    second_possible_chord = get_nth_key(ChordDict, n=np.where(prediction[0] == prediction_array_sorted[0][-2])[0][0])
    third_possible_chord = get_nth_key(ChordDict, n=np.where(prediction[0] == prediction_array_sorted[0][-3])[0][0])
    index = np.argmax(prediction)

    print(f"Possible Chords are: 1.{first_possible_chord}, 2.{second_possible_chord}, 3.{third_possible_chord}\n "
          f"with accuracy {prediction_array_sorted[0][-1]}, {prediction_array_sorted[0][-2]}, {prediction_array_sorted[0][-3]}")
    
def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")


    
model = create_network()
while True:
        Record()
        new_frequency=PrepareData(audiofile="test_recording.wav")
        New_freqArray=ProcessData(np.array(new_frequency))
        DoPredict(model, New_freqArray)
        if str(input("Predict again? 0 == exit: ")) == "0":
            break
