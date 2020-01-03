import numpy as np
np.random.seed(1158)  # for reproducibility
from image_pro import *
from keras.utils import np_utils
from keras.models import Sequential,load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Conv3D, MaxPooling3D,AveragePooling3D, Flatten,BatchNormalization
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import pandas as pd

#parameters
train_dir='train_val'
test_dir='test'
size=50

# training X shape (465, 50^3), Y shape (465, 1). test X shape (117, 50^3), Y shape (117,1)
(X_train,seg_train,y_train)=get_trainset(train_dir)
X_test,seg_test,test_files= get_testset(test_dir)

X_train = X_train.reshape(X_train.shape[0],size,size,size,1)
seg_train = seg_train.reshape(seg_train.shape[0],size,size,size,1)
X_test = X_test.reshape(X_test.shape[0],size,size,size,1)
seg_test = seg_test.reshape(seg_test.shape[0],size,size,size,1)
y_train = np_utils.to_categorical(y_train, num_classes=2)

# Another way to build your CNN
model = Sequential()
# Conv layer 1 output shape (32,34,34,34)
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Dropout(0.2))
model.add(Conv3D(
    32,
    kernel_dim1=17, # depth
    kernel_dim2=17, # rows
    kernel_dim3=17, # cols
    input_shape=(size,size,size,1),
    activation='relu',
))


# Conv layer 2 output shape (64,28,28,28)
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Conv3D(
        64,
        kernel_dim1=7, # depth
        kernel_dim2=7, # rows
        kernel_dim3=7, # cols
        activation='relu',
))

# Pooling layer 2 (average pooling) output shape (64,14,14,14)
model.add(MaxPooling3D(
    pool_size=(2,2,2)
))


# Conv layer 3 output shape (80,10,10,10)
model.add(BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
model.add(Conv3D(
        80,
        kernel_dim1=5, # depth
        kernel_dim2=5, # rows
        kernel_dim3=5, # cols
        activation='relu',
))

# Pooling layer 2 (average pooling) output shape (80,5,5,5)
model.add(MaxPooling3D(
    pool_size=(2,2,2)
))

# Fully connected layer 1 input shape 10000, output shape (128)
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

# Fully connected layer 3 to shape (2) for 2 classes

model.add(Dense(2))
model.add(Activation('softmax'))

# Another way to define your optimizer
adam = Adam(lr=1e-3)

# We add metrics to get more results you want to see
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training ------------')
# Another way to train the model
early_stopping=EarlyStopping(monitor='val_loss',patience=20,verbose=2)

best_keeper = ModelCheckpoint(filepath='tmp/test/best.h5', verbose=1,
                                  monitor='val_acc', save_best_only=True, period=1, mode='max')

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)


model.fit(X_train,y_train,validation_split=0.1,epochs=100, batch_size=20,callbacks=[early_stopping,best_keeper,lr_reducer])
#model.fit(seg_train,y_train,validation_split=0.1, epochs=100, batch_size=10,callbacks=[early_stopping,best_keeper,lr_reducer])

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
model = load_model('tmp/test/best.h5')
prediction = model.predict(X_test)

result=[]
for i in range(len(prediction)):
    result.append(float(prediction[i][1]))
Data = {'Id': test_files, 'Predicted': result}
DF=pd.DataFrame(Data, columns=["Id","Predicted"])

DF.to_csv('test.csv',encoding='utf-8')