# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:18:54 2022

@author: zjy52
"""
import os
import pandas as pd
import gc
from textwrap import dedent
from Bio import SeqIO
import numpy as np
import random
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Embedding, Activation, Input, concatenate, dot
from keras.layers import Conv1D, MaxPooling1D, LSTM, GRU
from keras.models import Sequential
from keras import regularizers
from keras.constraints import max_norm
from keras import initializers
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.optimizers import adam_v2
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
os.chdir('C:/Users/zjy52/Desktop/CNN-VF')
#TRAIN_DATA_READ
positive=[]
negative=[]
for seq in SeqIO.parse('dataset/po.1500.fa', 'fasta'):
    positive.append(str(seq.seq))
for seq_r in SeqIO.parse('dataset/ne.1500.fa', 'fasta'):
    negative.append(str(seq_r.seq))
#def_onehot
def one_hot_padding(seq_list,padding):
    feat_list = []
    one_hot = {}
    aa = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X','B','J','Z']
    for i in range(len(aa)):
        one_hot[aa[i]] = [0]*24
        one_hot[aa[i]][i] = 1 
    for i in range(len(seq_list)):
        feat = []
        for j in range(len(seq_list[i])):
            feat.append(one_hot[seq_list[i][j]])
        feat = feat + [[0]*24]*(padding-len(seq_list[i]))
        feat_list.append(feat)
        gc.collect()
    return(np.array(feat_list))
#predict_class
def predict_by_class(scores):
    classes = []
    for i in range(len(scores)):
        if scores[i]>0.5:
            classes.append(1)
        else:
            classes.append(0)
    return np.array(classes)
#model
def CNN_MODEL():
    maxlength=1500
    input_dim=24
    kernel_initia = initializers.random_normal(mean=0.0, stddev=0.05, seed=None)
    drop = 0.5
    n_filter = 128
    kernel = 7
    pool_size = 5
    model = Sequential()
    model.add(Conv1D(filters=n_filter, kernel_size=kernel, input_shape=(maxlength, input_dim,), name='conv1',
                         kernel_initializer=kernel_initia))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, name='maxpool2'))
    model.add(Conv1D(filters=n_filter, kernel_size=kernel, name='conv3', kernel_initializer=kernel_initia))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=pool_size, name='maxpool4'))
    model.add(Dropout(drop, name='dropout1'))
    model.add(Flatten(name='flatten5'))
    model.add(Dense(512, name='fl1', ))
    model.add(Dropout(drop, name='dropout2'))
    model.add(Dense(512, name='fl2', ))
    model.add(Dropout(drop, name='dropout3'))
    model.add(Dense(2, activation='softmax', name='prediction'))
    adam = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    return model
def plot_acc_loss(train_info_record, sig, train_values, val_values, epochs, ll=False):
    plt.clf()
    plt.plot(epochs, train_values, 'bo', label='Training')
    plt.plot(epochs, val_values, 'r', label='Validation')
    if ll:
        name = train_info_record + sig + '_' + str(ll)
        title = sig + '_' + 'training and validation ' + str(ll)
    else:
        name = train_info_record + sig
        title = sig + '_' + 'training and validation'
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(str(ll))
    plt.legend()
    plt.savefig(str(name) + '.png')
    plt.clf()
#data_deal
# sequences for training sets
train_seq = positive + negative    
# set labels for training sequences
y_train = np.array([1]*len(positive) + [0]*len(negative))
# shuffle training set
train = list(zip(train_seq, y_train))
random.Random(123).shuffle(train)
train_seq, y_train = zip(*train)
train_seq = list(train_seq)
y_train = np.array((y_train))
#data_distribution
X_train, X_test_a, Y_train,Y_test_a = train_test_split(train_seq, y_train, test_size=0.2, random_state=13)
X_test, X_val, Y_test, Y_val = train_test_split(X_test_a, Y_test_a, test_size=0.3, random_state=13)
del X_test_a, Y_test_a
print('train number is {}\ntest number is {}\n val number is {}\n'.format(len(X_train), len(X_test), len(X_val)))
#data_encode
Xtrain=one_hot_padding(X_train,1500)
Xval=one_hot_padding(X_val,1500)
Xtest=one_hot_padding(X_test, 1500)
Ytrain=to_categorical(Y_train)
Yval=to_categorical(Y_val)
#model_train
indv_pred_train = []
model=CNN_MODEL()
histroy=model.fit(Xtrain, Ytrain, epochs=100, validation_data=(Xval, Yval), batch_size=15)
temp_pred_train = model.predict(Xtrain).flatten()
temp_pred_class_train_curr = predict_by_class(model.predict(Xtrain).flatten())
temp_pred_class_val = predict_by_class(model.predict(Xval).flatten())
print('*************************** current model ***************************')
print('current train acc: ', accuracy_score(Ytrain, temp_pred_class_train_curr))
print('current val acc: ', accuracy_score(Yval, temp_pred_class_val))
#plot
f = open('train'+ '.txt', 'a')
loss_values=histroy.history['loss']
acc_values = histroy.history['accuracy']
val_loss_values = histroy.history['val_loss']
val_acc_values = histroy.history['val_accuracy']
epochs = range(1, len( acc_values) + 1)
for i in range(0,  len(acc_values)):
    f.write('epoch:{}, train_loss:{:.4f}, train_acc:{:.4f}, val_loss:{:.4f}, val_acc:{:.4f}'.
            format(i + 1, loss_values[i], acc_values[i], val_loss_values[i], val_acc_values[i]) + '\n')
plot_acc_loss('VF', 'train', acc_values, val_acc_values, epochs, ll='acc')  # plot acc
plot_acc_loss('VF', 'val', loss_values, val_acc_values, epochs, ll='acc')  #  plot loss
f.close()
#test
#te_postive=[]
#te_negative=[]
#for te_seq in SeqIO.parse('data/AMP_test.fa', 'fasta'):
#    te_postive.append(str(te_seq.seq))
#for te_seq2 in SeqIO.parse('data/non_AMP_test.fa','fasta'):
 #   te_negative.append(str(te_seq2.seq))
#test_seq=te_postive+te_negative
#y_test=np.array([1]*len(te_postive)+[0]*len(te_negative))
#test_train=list(zip(test_seq,y_test))
#random.Random(123).shuffle(test_train)
#model_test,y_test=zip(*test_train)
#model_test=list(model_test)
#y_test=np.array(y_test)
#model_test=one_hot_padding(model_test, 200)
#test_result=model.predict(model_test).flatten()
#test_curr=predict_by_class(test_result)
#print('********************test_result*********************')
#print('current test acc:', accuracy_score(y_test, test_curr))

