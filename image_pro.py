import numpy as np
import pandas as pd
import os

def get_trainset(train_dir):
    X_train=[]
    seg_train=[]
    y_train=[]
    filelists = os.listdir(train_dir)
    sort_num_first = []
    for file in filelists:
        tes=''
        i=9
        while file[i]!='.':
            tes=tes+file[i]
            i=i+1
        sort_num_first.append(int(tes))   
    sort_num_first.sort()

    sorted_files = []
    for sort_num in sort_num_first:
        sorted_files.append('candidate'+str(sort_num)+'.npz')
            
    for file in sorted_files:
        tmp=np.load(train_dir+'/'+file)
        temp1=tmp['voxel'][25:75,25:75,25:75]
        temp2=tmp['seg'][25:75,25:75,25:75]
        X_train.append(temp1/255.)
        seg_train.append(temp2/255.)
    X_train=np.array(X_train)
    seg_train=np.array(seg_train)
    csv_data=pd.read_csv('train_val.csv')
    data=csv_data['lable']

    for i in range(len(data)):
        y_train.append(np.array([data[i]]))
    y_train=np.array(y_train)

    return X_train,seg_train,y_train

def get_testset(test_dir):
    X_test=[]
    seg_test=[]
    filelists = os.listdir(test_dir)
    sort_num_first = []
    for file in filelists:
        tes=''
        i=9
        while file[i]!='.':
            tes=tes+file[i]
            i=i+1
        sort_num_first.append(int(tes))   
    sort_num_first.sort()

    sorted_files = []
    sorted_names=[]
    for sort_num in sort_num_first:
        sorted_names.append('candidate'+str(sort_num))
        sorted_files.append('candidate'+str(sort_num)+'.npz')

    for file in sorted_files:
        tmp=np.load(test_dir+'/'+file)
        temp1=tmp['voxel'][25:75,25:75,25:75]
        temp2=tmp['seg'][25:75,25:75,25:75]
        X_test.append(temp1/255.)
        seg_test.append(temp2/255.)
    X_test=np.array(X_test)
    seg_test=np.array(seg_test)

    return X_test,seg_test,sorted_names