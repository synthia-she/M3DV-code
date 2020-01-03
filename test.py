import numpy as np
np.random.seed(1158)  # for reproducibility
from image_pro import *
from keras.utils import np_utils
from keras.models import Sequential,load_model,Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Conv3D, MaxPooling3D,AveragePooling3D, Flatten,BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import pandas as pd

#parameters
model_path='model.h5'
test_dir='test'
size=50

X_test,seg_test,test_files= get_testset(test_dir)

X_test = X_test.reshape(X_test.shape[0],size,size,size,1)
seg_test = seg_test.reshape(seg_test.shape[0],size,size,size,1)
 
model=load_model(model_path)
prediction = model.predict(X_test)

result=[]
for i in range(len(prediction)):
    result.append(float(prediction[i][1]))
Data = {'Id': test_files, 'Predicted': result}
DF=pd.DataFrame(Data, columns=["Id","Predicted"])

DF.to_csv('submission.csv',encoding='utf-8',index = False)