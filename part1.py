import librosa
import os
import pandas as pd
import librosa
import glob
import librosa.display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn import metrics 


def parser(row):
   # function to load files and extract features
   file_name = os.path.join(os.path.abspath('./data'), 'Train', str(row.ID) + '.wav')

   # handle exception to check if there isn't a file which is corrupted
   try:
      # here kaiser_fast is a technique used for faster extraction
      X, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
      # we extract mfcc feature from data
      mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
   except Exception as e:
      print("Error encountered while parsing file: ", file)
      return None, None
 
   feature = mfccs
   label = row.Class
 
   return [feature, label]

df = None

if os.path.exists('./urban/input.csv'):
   df = pd.read_csv('./urban/input.csv')
   df['feature']=df['feature'].map(lambda s: list(map(float,s.strip(' []').split())))
   print('load file')
else:
   print('parse file')
   train = pd.read_csv(os.path.join('./urban', 'train.csv'))
   temp = train.apply(parser, axis=1)
   df=pd.DataFrame(temp.to_list(),columns = ['feature','label'])

X = np.array(df.feature.tolist())
y = np.array(df.label.tolist())
lb = LabelEncoder()
y = np_utils.to_categorical(lb.fit_transform(y))

num_labels = y.shape[1]

def make_Model():
    filter_size = 2
    # build model
    model = Sequential()

    model.add(Dense(256, input_shape=(40,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    return model

def train_Model(split):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, random_state=22)
    model = make_Model()
    model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))
    return model

print(y)
print(y.shape)

# model1 = train_Model(0.2)
# model2 = train_Model(0.5)
