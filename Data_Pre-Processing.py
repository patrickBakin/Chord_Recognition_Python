import librosa
import matplotlib.pyplot as plt
import math
import numpy as np
import glob
import soundfile
import os
import shutil
from audiomentations import Compose,AddGaussianNoise,TimeStretch,PitchShift,Shift


augment = Compose([
        AddGaussianNoise(min_amplitude=0.0008, max_amplitude=0.001, p=1),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

def createFolder():
    foldername=[]
    for file in glob.glob('data/audio_augmented_x10/*.wav'):
        print("parsing file %s" %file)
        filenamelist = str(file).split('_')
        
        if filenamelist[4]+"_"+filenamelist[5] not in foldername:
            foldername.append(filenamelist[4]+"_"+filenamelist[5])
            os.mkdir(f'data\\audio_augmented_x10\\{foldername[-1]}')

    print(foldername)



def AugmentSound():

    folder = [dI for dI in os.listdir('data\\audio_augmented_x10\\') if
              os.path.isdir(os.path.join('data\\audio_augmented_x10\\', dI))]
    for i in range(48):
            #Key=get_nth_key(ChordDict, n=i)
            for file in glob.glob(f'data\\audio_augmented_x10\\{folder[i]}\\*.wav'):
                    f,sr =librosa.load(file,sr=None)
                    augmented_audio=augment(f,sr)
                    soundfile.write(file,augmented_audio,sr)
                    print("Augmenting "+str(file))

def moveFiles():
    folder = [dI for dI in os.listdir('data\\audio_augmented_x10\\') if
              os.path.isdir(os.path.join('data\\audio_augmented_x10\\', dI))]

    for file in glob.glob('data\\audio_augmented_x10\\*.wav'):
        filenamelist = str(file).split('_')
        foldername=filenamelist[4]+"_"+filenamelist[5]
        if foldername in folder:
            destpath =  f"data\\audio_augmented_x10\\{foldername}\\"+getFileName(file)
            shutil.move(file,destpath)
            print("moving "+getFileName(file))


def moveValFilesToNewFolder():
    folder = [dI for dI in os.listdir('data\\audio_augmented_x10\\') if
              os.path.isdir(os.path.join('data\\audio_augmented_x10\\', dI))]
    for foldername in folder:
        os.mkdir(f'data\\audio_augmented_x10_val\\{foldername}')

    for i in range(48):
            #Key=get_nth_key(ChordDict, n=i)
            for file in glob.glob(f'data\\audio_augmented_x10\\{folder[i]}\\*.wav'):


                if int(str(file).split('\\')[3].split("_")[5].split(".")[0]) >=80 and str(file).split('\\')[3].split("_")[5] !="00.wav":
                    destpath = f"data\\audio_augmented_x10_val\\{folder[i]}\\"+str(file).split('\\')[3]
                    shutil.move(file,destpath)
                    print("Moving "+file)
                    print("Destination Path "+destpath)

def getFileName(string):
    return string.split('\\')[2]

def get_nth_key(dictionary, n=0):
    if n < 0:
        n += len(dictionary)
    for i, key in enumerate(dictionary.keys()):
        if i == n:
            return key
    raise IndexError("dictionary index out of range")

def Process(ValidationSet=False):

    if ValidationSet == True:

        folderpath = "audio_augmented_x10_val"
        filename = "ChordData_Val.npz"
    else:
        folderpath = "audio_augmented_x10"
        filename = "ChordData.npz"


    folder = [dI for dI in os.listdir('data\\audio_augmented_x10\\') if
              os.path.isdir(os.path.join('data\\audio_augmented_x10\\', dI))]

    ChordDict={'A Flat Aug':[],'A Flat Dim':[],'A Flat Maj':[],'A Flat Min':[],
              'A Aug': [], 'A Dim': [], 'A  Maj': [], 'A  Min': [],
              'B Flat Aug':[],'B Flat Dim':[],'B Flat Maj':[],'B Flat Min':[],
              'B Aug': [], 'B Dim': [], 'B  Maj': [], 'B  Min': [],
              'C Aug':[],'C Dim':[],'C Maj':[],'C Min':[],
              'D Flat Aug':[],'D Flat Dim':[],'D Flat Maj':[],'D Flat Min':[],
              'D Aug': [], 'D Dim': [], 'D  Maj': [], 'D  Min': [],
              'E Flat Aug':[],'E Flat Dim':[],'E Flat Maj':[],'E Flat Min':[],
              'E Aug': [], 'E Dim': [], 'E Maj': [], 'E  Min': [],
              'F Aug':[],'F Dim':[],'F Maj':[],'F Min':[],
              'G Flat Aug':[],'G Flat Dim':[],'G Flat Maj':[],'G Flat Min':[],
              'G Aug': [], 'G Dim': [], 'G  Maj': [], 'G  Min': []}

 
    Chordlist=[]

    for i in range(48):
            Key=get_nth_key(ChordDict, n=i)
            for file in glob.glob(f'data\\{folderpath}\\{folder[i]}\\*.wav'):

                print("Storing "+str(file))
                newSample = Resampling(file)
                ChordDict[Key].append(newSample)

            Chordlist.append(np.array(ChordDict[Key]))
            ChordDict[Key]=[]

    ChordArray=np.array(Chordlist)

    print("Array Shape: ",ChordArray.shape)
    np.savez_compressed(filename,a=ChordArray)
    
def Resampling(wave):

    f, sr = librosa.load(wave, sr=None,offset=0.0, duration=0.9)

    output = librosa.resample(f, orig_sr=sr, target_sr=16000)

    st = librosa.stft(output, n_fft=1024, hop_length=256)

 
    spectogram = np.abs(st) ** 2
   
    spectogram = spectogram.round(decimals=5)

    return spectogram

if __name__ == '__main__':

    createFolder()
    moveFiles()
    moveValFilesToNewFolder()
    AugmentSound()
    Process(ValidationSet=False)
    Process(ValidationSet=True)

