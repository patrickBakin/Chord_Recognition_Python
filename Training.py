from keras.models import Sequential
import tensorflow as tf
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import train_test_split
import os
from keras.layers import Dense,MaxPooling2D,Conv2D,Flatten
from keras.utils import np_utils

def getInOutarray(npzfile):

    #
    loaded_data=np.load(npzfile)

    data=loaded_data['a']
    data_shape1=data.shape[1]
    input_arr =np.reshape(data,(data.shape[0]*data.shape[1],data.shape[2],data.shape[3],1))
    input_arr = input_arr/float(9957.391)
    output_arr=np.array([])

    for i in range(48):
        a = np.array([i])
        b = np.repeat(a,repeats=data_shape1,axis=0)
        output_arr=np.append(output_arr,b)


    array_output = np_utils.to_categorical(output_arr)

    return input_arr,array_output



def create_network(n_inputs,n_outputs):

    model = Sequential()
    model.add(Conv2D(128,(3,3),activation='relu',input_shape=n_inputs.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Dense(16))
    model.add(Dense(8))
    model.add(Flatten())
    model.add(Dense(48,activation="softmax"))
    adm = tf.keras.optimizers.Adam(learning_rate=0.001) #0.003
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    
    print(model.summary())

    return model

def train(model,X_train, X_test, Y_train, Y_test):

    filepath = os.path.abspath("weights-Chord-{epoch:03d}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5")
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, cooldown=1,
                                                     patience=4, min_lr=0.0001)
    callback_list =[checkpoint,reduce_lr]
    model.fit(X_train,Y_train,validation_data=[X_test,Y_test],epochs=1000,batch_size=8,verbose=1,callbacks=callback_list)


if __name__ == '__main__':

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    X_train,Y_train= getInOutarray('ChordData.npz')
    X_test,Y_test =getInOutarray('ChordData_val.npz')
    model = create_network(X_train,Y_train)
    train(model,X_train, X_test, Y_train, Y_test)



