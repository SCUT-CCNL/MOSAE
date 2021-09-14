# -*- coding: utf-8 -*-
"""
Created on Sat May 18 15:24:37 2019

@author: kevin
"""

import pandas as pd
import numpy as np
from scipy import stats

from sklearn import svm, tree, naive_bayes, neighbors, ensemble
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

import tensorflow as tf
from keras import layers, regularizers, optimizers
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K

def _compute_fans(shape, data_format='channels_last'):
    """Computes the number of input and output units for a weight shape.

    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).

    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.

    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def rnaseq_preprecess(rnaseq):
    index_temp=rnaseq.index
    rnaseq=rnaseq.T
    rnaseq=rnaseq[(rnaseq>8).sum(axis=1)!=0]
    rnaseq=rnaseq[((rnaseq==0).sum(axis=1)==0)]
    rnaseq=np.log2(rnaseq)

    rnaseq=rnaseq.T
    rnaseq=StandardScaler().fit_transform(rnaseq)
    rnaseq=pd.DataFrame(rnaseq)
    rnaseq.index=index_temp
    rnaseq=rnaseq.T.dropna().T
    return rnaseq

# def remove_outlier_zscore(df):
#     df=df.T
#     df=df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
#     return df.T

def remove_outlier_IQR(df):
    df=df.T
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df.T

def preprocess_somatic(df):
    df=df.T
    df=df[(df==0).sum(axis=1)!=df.shape[1]]
    return df.T

def preprocess_methy(df):
    df=df.T
    df=df[(df.std(axis=1)>0.15)]
    df=df[(df.mean(axis=1)>0.2)]
    return df.T

def preprocess_rnaseq(df):
    # df=remove_outlier_zscore(df)
    df=df.T
    df=df[(df.std(axis=1)>1)]
    df=df[(df.mean(axis=1)>6)]
    return df.T

def preprocess_cnv(df):
    df=df.T
    df=df[df.std(axis=1)>0.2]
    df=df[np.abs(df.mean(axis=1))>0.05]
    return df.T

def load_data_os():
    print('loading OS data...')

    # methy=pd.read_table('../../dataset/Pan-os_pfi/methy.txt', index_col=0)
    # rnaseq=pd.read_table('../../dataset/Pan-os_pfi/rnaseq.txt', index_col=0)
    # mirna=pd.read_table('../../dataset/Pan-os_pfi/mirna.txt', index_col=0)
    # rppa=pd.read_table('../../dataset/Pan-os_pfi/rppa.txt', index_col=0)
    # clin=pd.read_table('../../dataset/Pan-os_pfi/clin.txt', index_col=0)
    #
    # methy=preprocess_methy(methy)
    # rnaseq=preprocess_rnaseq(rnaseq)
    # # mirna=remove_outlier_zscore(mirna)
    #
    # #cnv.columns=['cnv']*cnv.shape[1]
    # methy.columns=['methy']*methy.shape[1]
    # rnaseq.columns=['rnaseq']*rnaseq.shape[1]
    # mirna.columns=['mirna']*mirna.shape[1]
    # rppa.columns=['rppa']*rppa.shape[1]
    # #somatic.columns=['somatic']*somatic.shape[1]
    #
    # sample=pd.concat([methy, rnaseq, mirna, rppa], axis=1)
    DATASET = "OS"
    sample = pd.read_table("../../dataset/" + DATASET + ".txt", index_col=0)
    clin = pd.read_table("../../dataset/" + DATASET + "_clin.txt")
    target_os=clin[DATASET].values
    return sample, target_os

def load_data_pfi():
    print('loading PFI data...')

    # methy=pd.read_table('dataset/Pan-os_pfi/methy.txt', index_col=0)
    # rnaseq=pd.read_table('dataset/Pan-os_pfi/rnaseq.txt', index_col=0)
    # mirna=pd.read_table('dataset/Pan-os_pfi/mirna.txt', index_col=0)
    # rppa=pd.read_table('dataset/Pan-os_pfi/rppa.txt', index_col=0)
    # clin=pd.read_table('dataset/Pan-os_pfi/clin.txt', index_col=0)
    #
    # methy=preprocess_methy(methy)
    # rnaseq=preprocess_rnaseq(rnaseq)
    # # mirna=remove_outlier_zscore(mirna)
    #
    # #cnv.columns=['cnv']*cnv.shape[1]
    # methy.columns=['methy']*methy.shape[1]
    # rnaseq.columns=['rnaseq']*rnaseq.shape[1]
    # mirna.columns=['mirna']*mirna.shape[1]
    # rppa.columns=['rppa']*rppa.shape[1]
    # #somatic.columns=['somatic']*somatic.shape[1]
    #
    # sample=pd.concat([methy, rnaseq, mirna, rppa], axis=1)
    # target_pfi=clin['PFI'].values
    # return sample, target_pfi
    DATASET = "PFI"
    sample = pd.read_table("../../dataset/" + DATASET + ".txt", index_col=0)
    clin = pd.read_table("../../dataset/" + DATASET + "_clin.txt")
    target_os=clin[DATASET].values
    return sample, target_os

def load_data_dfi():
    print('loading DFI data...')
    # methy=pd.read_table('dataset/Pan-dfi/methy.txt', index_col=0)
    # rnaseq=pd.read_table('dataset/Pan-dfi/rnaseq.txt', index_col=0)
    # mirna=pd.read_table('dataset/Pan-dfi/mirna.txt', index_col=0)
    # rppa=pd.read_table('dataset/Pan-dfi/rppa.txt', index_col=0)
    # clin=pd.read_table('dataset/Pan-dfi/clin.txt', index_col=0)
    #
    #
    # methy=preprocess_methy(methy)
    # rnaseq=preprocess_rnaseq(rnaseq)
    # # mirna=remove_outlier_zscore(mirna)
    #
    # #cnv.columns=['cnv']*cnv.shape[1]
    # methy.columns=['methy']*methy.shape[1]
    # rnaseq.columns=['rnaseq']*rnaseq.shape[1]
    # mirna.columns=['mirna']*mirna.shape[1]
    # rppa.columns=['rppa']*rppa.shape[1]
    # #somatic.columns=['somatic']*somatic.shape[1]
    #
    # sample=pd.concat([methy, rnaseq, mirna, rppa], axis=1)
    # target_dfs=clin['DFI'].values
    # del methy, rnaseq, mirna, rppa, clin
    # return sample, target_dfs
    DATASET = "DFI"
    sample = pd.read_table("../../dataset/" + DATASET + ".txt", index_col=0)
    clin = pd.read_table("../../dataset/" + DATASET + "_clin.txt")
    target_os=clin[DATASET].values
    return sample, target_os

def load_data_dss():
    print('loading DSS data...')
    # methy=pd.read_table('dataset/Pan-dss/methy.txt', index_col=0)
    # rnaseq=pd.read_table('dataset/Pan-dss/rnaseq.txt', index_col=0)
    # mirna=pd.read_table('dataset/Pan-dss/mirna.txt', index_col=0)
    # rppa=pd.read_table('dataset/Pan-dss/rppa.txt', index_col=0)
    # clin=pd.read_table('dataset/Pan-dss/clin.txt', index_col=0)
    #
    #
    # methy=preprocess_methy(methy)
    # rnaseq=preprocess_rnaseq(rnaseq)
    # # mirna=remove_outlier_zscore(mirna)
    #
    # #cnv.columns=['cnv']*cnv.shape[1]
    # methy.columns=['methy']*methy.shape[1]
    # rnaseq.columns=['rnaseq']*rnaseq.shape[1]
    # mirna.columns=['mirna']*mirna.shape[1]
    # rppa.columns=['rppa']*rppa.shape[1]
    # #somatic.columns=['somatic']*somatic.shape[1]
    #
    # sample=pd.concat([methy, rnaseq, mirna, rppa], axis=1)
    # target_dfs=clin['DSS'].values
    # del methy, rnaseq, mirna, rppa, clin
    # return sample, target_dfs
    DATASET = "DSS"
    sample = pd.read_table("../../dataset/" + DATASET + ".txt", index_col=0)
    clin = pd.read_table("../../dataset/" + DATASET + "_clin.txt")
    target_os=clin[DATASET].values
    return sample, target_os

def load_data(targettype):
    if targettype == 'OS':
        sample, target = load_data_os()
    elif targettype == 'DSS':
        sample, target = load_data_dss()
    elif targettype == 'PFI':
        sample, target = load_data_pfi()
    elif targettype == 'DFI':
        sample, target = load_data_dfi()

    return sample, target

#clfs = [
#    ('SVC', svm.SVC(kernel='linear', probability=True)),
#    ('DT', tree.DecisionTreeClassifier()),
#    ('NB', naive_bayes.GaussianNB()),
#    ('kNN', neighbors.KNeighborsClassifier(n_jobs=-1)),
#    ('RF', ensemble.RandomForestClassifier(n_jobs=-1)),
#    ('AB',ensemble.AdaBoostClassifier())
#]

##论文里的baseline算法
clfs = [('SVC', svm.SVC(kernel="linear", C=0.025)),
              ('DT', tree.DecisionTreeClassifier(max_depth=5)),
              ('NB', naive_bayes.GaussianNB()),
              ('kNN', neighbors.KNeighborsClassifier(5)),
              ('RF', ensemble.RandomForestClassifier(max_depth=5, n_estimators=10)),
              ('AB', ensemble.AdaBoostClassifier())
              ]

#datasets = [
#    ('OS', load_data_os())
#    ('PFI', load_data_pfi())
#    ('DFO', load_data_dfi())
#]
def my_init(shape, dtype=None):
    fan_in, _ = _compute_fans(shape)
    limit = np.sqrt(1. / fan_in)
    return K.random_uniform(shape, -limit, limit,
                            dtype=dtype, seed=1)
def SAE(X_train, X_test, dim, epo=30, bs=8):
    print('SAEing...')
    #结构
    input_data = Input(shape=(X_train.shape[1],))
    encoded = Dense(dim, activation = 'tanh',
                   kernel_regularizer = regularizers.l1(0.0001),
                    activity_regularizer=regularizers.l2(0.001))(input_data)
    decoded = Dense(X_train.shape[1], activation = 'linear')(encoded)

    autoencoder = Model(input_data, decoded)
    encoder = Model(input_data, encoded)
    #优化和损失函数
    sgd = optimizers.SGD(lr=0.1, clipvalue=0.05)
    autoencoder.compile(optimizer=sgd, loss='mean_squared_error', metrics=['accuracy'])
    #训练
    autoencoder.fit(X_train, X_train,
                epochs=epo,
                batch_size=bs,
                shuffle=True,
                verbose=0)
    return encoder.predict(X_train), encoder.predict(X_test)

def AE_T(X_train, X_test, y_train, y_test, dim=100, epo=100, bs=1000):
    print('AE_Ting...')
    #结构
    input_data = Input(shape=(X_train.shape[1],))
    h1=Dense(dim, activation = 'relu')(input_data)
    encoded = Dense(dim, activation = 'relu')(h1)
    decoded = Dense(X_train.shape[1], activation = 'linear', name='ctg_out_1')(encoded)

    predicted =Dense(1, activation = 'sigmoid', name='ctg_out_2')(encoded)

    autoencoder = Model(input_data, outputs=[decoded, predicted])
    #encoder = Model(input_data, encoded)
    predictor = Model(input_data, predicted)
    #autoencoder.summary()#200w
    #predictor.summary()#100w
    #优化和损失函数
    autoencoder.compile(optimizer='SGD', loss={'ctg_out_1': 'mean_squared_error', 'ctg_out_2': 'binary_crossentropy'},
                        loss_weights={ 'ctg_out_1': 1.,'ctg_out_2': 1.})
    #训练
    autoencoder.fit(X_train, [X_train, y_train],
                epochs=epo,
                batch_size=bs,
                shuffle=True,
                verbose=0)
    plot_model(autoencoder, to_file='AE_T.png', show_shapes=True)
    train_roc=roc_auc_score(y_train, predictor.predict(X_train))
    test_roc=roc_auc_score(y_test, predictor.predict(X_test))
    print(train_roc)
    print(test_roc)
    return test_roc

def AE_T_M_F(X_train, X_test, y_train, y_test, dim=100, epo=100, bs=1000):
    print('AE_T_M_Fing...')
    #结构
    input_data_m = Input(shape=(X_train['methy'].shape[1],))
    encoded_m = Dense(dim, activation = 'relu')(input_data_m)
    decoded_m = Dense(X_train['methy'].shape[1], activation = 'linear', name='methy')(encoded_m)

    input_data_r = Input(shape=(X_train['rnaseq'].shape[1],))
    encoded_r = Dense(dim, activation = 'relu')(input_data_r)
    decoded_r = Dense(X_train['rnaseq'].shape[1], activation = 'linear', name='rnaseq')(encoded_r)

    input_data_mi = Input(shape=(X_train['mirna'].shape[1],))
    encoded_mi = Dense(dim, activation = 'relu')(input_data_mi)
    decoded_mi = Dense(X_train['mirna'].shape[1], activation = 'linear', name='mirna')(encoded_mi)

    input_data_rp = Input(shape=(X_train['rppa'].shape[1],))
    encoded_rp = Dense(dim, activation = 'relu')(input_data_rp)
    decoded_rp = Dense(X_train['rppa'].shape[1], activation = 'linear', name='rppa')(encoded_rp)

    cat_layer = layers.average([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #sum_layer = layers.add([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #fusion_layer = Dense(dim, activation = 'relu')(sum_layer)
    predicted =Dense(1, activation = 'sigmoid', name='predictor')(cat_layer)

    autoencoder = Model([input_data_m, input_data_r, input_data_mi, input_data_rp],
                        outputs=[decoded_m, decoded_r, decoded_mi, decoded_rp, predicted])
    #encoder = Model(input_data, encoded)
    predictor = Model([input_data_m, input_data_r, input_data_mi, input_data_rp], predicted)

    #autoencoder.summary()#200w
    #predictor.summary()#100w
    #优化和损失函数
    autoencoder.compile(optimizer='SGD',
                        loss={'methy': 'mean_squared_error',
                              'rnaseq': 'mean_squared_error',
                              'mirna': 'mean_squared_error',
                              'rppa': 'mean_squared_error',
                              'predictor': 'binary_crossentropy'},
                        loss_weights={'methy': 1.,
                                      'rnaseq': 1.,
                                      'mirna': 1.,
                                      'rppa': 1.,
                                      'predictor': 1.})
    #训练
    autoencoder.fit([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']],
                    [X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa'],
                     y_train],
                    epochs=epo,
                    batch_size=bs,
                    shuffle=True,
                    verbose=0)

    plot_model(autoencoder, to_file='AE_T_M_F_ave.png', show_shapes=True)

    train_roc=roc_auc_score(y_train, predictor.predict([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']]))
    test_roc=roc_auc_score(y_test, predictor.predict([X_test['methy'], X_test['rnaseq'], X_test['mirna'], X_test['rppa']]))

    print('result:', train_roc)
    print('result', test_roc)
    return test_roc

def AE_T_M_F_MT(X_train, X_test, y_train, y_test, dim=100, epo=100, bs=1000):
    ### MOSAE
    # AE: autoencoder;
    # T: final target for fused representation;
    # M: multi-view;
    # F: fuse by average;
    # MT: multi-targets, each for a single omics

    print('AE_T_M_F_MTing...')
    #结构
    dim_max = max([X_train['methy'].shape[1], X_train['rnaseq'].shape[1], X_train['mirna'].shape[1], X_train['rppa'].shape[1]])

    input_data_m = Input(shape=(X_train['methy'].shape[1],))
    h1_m = Dense(dim_max, activation = 'relu', kernel_initializer=my_init)(input_data_m)
    encoded_m = Dense(dim, activation = 'relu', kernel_initializer=my_init)(h1_m)
    decoded_m = Dense(X_train['methy'].shape[1], activation = 'linear', name='methy', kernel_initializer=my_init)(encoded_m)
    predicted_m =Dense(1, activation = 'sigmoid', name='predictor_methy', kernel_initializer=my_init)(encoded_m)

    input_data_r = Input(shape=(X_train['rnaseq'].shape[1],))
    h1_r = Dense(dim_max, activation = 'relu', kernel_initializer=my_init)(input_data_r)
    encoded_r = Dense(dim, activation = 'relu', kernel_initializer=my_init)(h1_r)
    decoded_r = Dense(X_train['rnaseq'].shape[1], activation = 'linear', name='rnaseq', kernel_initializer=my_init)(encoded_r)
    predicted_r = Dense(1, activation = 'sigmoid', name='predictor_rnaseq', kernel_initializer=my_init)(encoded_r)

    input_data_mi = Input(shape=(X_train['mirna'].shape[1],))
    h1_mi = Dense(dim_max, activation = 'relu', kernel_initializer=my_init)(input_data_mi)
    encoded_mi = Dense(dim, activation = 'relu', kernel_initializer=my_init)(h1_mi)
    decoded_mi = Dense(X_train['mirna'].shape[1], activation = 'linear', name='mirna', kernel_initializer=my_init)(encoded_mi)
    predicted_mi = Dense(1, activation = 'sigmoid', name='predictor_mirna', kernel_initializer=my_init)(encoded_mi)

    input_data_rp = Input(shape=(X_train['rppa'].shape[1],))
    h1_rp = Dense(dim_max, activation = 'relu', kernel_initializer=my_init)(input_data_rp)
    encoded_rp = Dense(dim, activation = 'relu', kernel_initializer=my_init)(h1_rp)
    decoded_rp = Dense(X_train['rppa'].shape[1], activation = 'linear', name='rppa', kernel_initializer=my_init)(encoded_rp)
    predicted_rp =Dense(1, activation = 'sigmoid', name='predictor_rppa', kernel_initializer=my_init)(encoded_rp)

    avg_layer = layers.average([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #cat_layer = layers.concatenate([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #sum_layer = layers.add([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #fusion_layer = Dense(dim, activation = 'relu')(sum_layer)
    predicted =Dense(1, activation = 'sigmoid', name='predictor', kernel_initializer=my_init)(avg_layer)

    autoencoder = Model([input_data_m, input_data_r, input_data_mi, input_data_rp],
                        outputs=[decoded_m, decoded_r, decoded_mi, decoded_rp, predicted_m, predicted_r, predicted_mi, predicted_rp, predicted])
    #autoencoder = Model([input_data_m, input_data_r, input_data_mi, input_data_rp],
      #                  outputs=[decoded_m, decoded_r, predicted_m, predicted_r, predicted_mi, predicted_rp, predicted])
    predictor_m = Model(input_data_m, predicted_m)
    predictor_r = Model(input_data_r, predicted_r)
    predictor_mi = Model(input_data_mi, predicted_mi)
    predictor_rp = Model(input_data_rp, predicted_rp)
    predictor = Model([input_data_m, input_data_r, input_data_mi, input_data_rp], predicted)

    #autoencoder.summary()#200w
    #predictor.summary()#100w
    #优化和损失函数
    autoencoder.compile(optimizer='SGD',
                        loss={'methy': 'mean_squared_error',
                              'rnaseq': 'mean_squared_error',
                              'mirna': 'mean_squared_error',
                              'rppa': 'mean_squared_error',
                              'predictor_methy': 'binary_crossentropy',
                              'predictor_rnaseq': 'binary_crossentropy',
                              'predictor_mirna': 'binary_crossentropy',
                              'predictor_rppa': 'binary_crossentropy',
                              'predictor': 'binary_crossentropy'},
                        loss_weights={'methy': 1.,
                                      'rnaseq': 1.,
                                      'mirna': 1.,
                                      'rppa': 1.,
                                      'predictor_methy': 1.,
                                      'predictor_rnaseq': 1.,
                                      'predictor_mirna': 1.,
                                      'predictor_rppa': 1.,
                                      'predictor': 1.})
    #训练
    autoencoder.fit([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']],
                    [X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa'],
                     y_train, y_train, y_train, y_train, y_train],
                    epochs=epo,
                    batch_size=X_train.shape[0],
                    shuffle=True,
                    verbose=1)

    plot_model(autoencoder, to_file='AE_T_M_F_MT_1layer_avg.png', show_shapes=True)
    train_roc_methy=roc_auc_score(y_train, predictor_m.predict(X_train['methy']))
    test_roc_methy=roc_auc_score(y_test, predictor_m.predict(X_test['methy']))

    train_roc_rnaseq=roc_auc_score(y_train, predictor_r.predict(X_train['rnaseq']))
    test_roc_rnaseq=roc_auc_score(y_test, predictor_r.predict(X_test['rnaseq']))

    train_roc_mirna=roc_auc_score(y_train, predictor_mi.predict(X_train['mirna']))
    test_roc_mirna=roc_auc_score(y_test, predictor_mi.predict(X_test['mirna']))

    train_roc_rppa=roc_auc_score(y_train, predictor_rp.predict(X_train['rppa']))
    test_roc_rppa=roc_auc_score(y_test, predictor_rp.predict(X_test['rppa']))

    train_roc=roc_auc_score(y_train, predictor.predict([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']]))
    test_roc=roc_auc_score(y_test, predictor.predict([X_test['methy'], X_test['rnaseq'], X_test['mirna'], X_test['rppa']]))

    print('result_methy_train:', train_roc_methy)
    print('result_methy_test', test_roc_methy)
    print('result_rnaseq_train:', train_roc_rnaseq)
    print('result_rnaseq_test', test_roc_rnaseq)
    print('result_mirna_train:', train_roc_mirna)
    print('result_mirna_test', test_roc_mirna)
    print('result_rppa_train:', train_roc_rppa)
    print('result_rppa_test', test_roc_rppa)
    print('result:', train_roc)
    print('result', test_roc)
    return test_roc_methy, test_roc_rnaseq, test_roc_mirna, test_roc_rppa, test_roc



def NN_Sequential(X_train, X_test, y_train, y_test, dim=100, epo=100, bs=1000):
    print('NN_Sequentialing...')
    #结构
    input_data = Input(shape=(X_train.shape[1],))
    h1=Dense(1000, activation = 'relu', kernel_initializer=my_init)(input_data)
    encoded = Dense(100, activation = 'relu', kernel_initializer=my_init)(h1)
    predicted =Dense(1, activation = 'sigmoid', kernel_initializer=my_init)(encoded)
    predictor = Model(input_data, predicted)
    #优化和损失函数
    predictor.compile(optimizer='SGD', loss='binary_crossentropy')
    #predictor.summary()#100w
    #训练
    bs=X_train.shape[0]
    predictor.fit(X_train, y_train,
                epochs=epo,
                batch_size=bs,
                shuffle=True,
                verbose=1)
    plot_model(predictor, to_file='NN_Sequential.png', show_shapes=True)
    train_roc=roc_auc_score(y_train, predictor.predict(X_train))
    test_roc=roc_auc_score(y_test, predictor.predict(X_test))
    print(train_roc)
    print(test_roc)
    return test_roc

def M_NN_Sequential(X_train, X_test, y_train, y_test, dim=100, epo=100, bs=1000):
    print('M_NN_Sequentialing...')
    #结构
    dim_max = max([X_train['methy'].shape[1], X_train['rnaseq'].shape[1], X_train['mirna'].shape[1], X_train['rppa'].shape[1]])

    input_data_m = Input(shape=(X_train['methy'].shape[1],))
    encoded_m = Dense(dim, activation = 'relu')(input_data_m)

    input_data_r = Input(shape=(X_train['rnaseq'].shape[1],))
    encoded_r = Dense(dim, activation = 'relu')(input_data_r)

    input_data_mi = Input(shape=(X_train['mirna'].shape[1],))
    encoded_mi = Dense(dim, activation = 'relu')(input_data_mi)

    input_data_rp = Input(shape=(X_train['rppa'].shape[1],))
    encoded_rp = Dense(dim, activation = 'relu')(input_data_rp)

    avg_layer = layers.average([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #cat_layer = layers.concatenate([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #sum_layer = layers.add([encoded_m, encoded_r, encoded_mi, encoded_rp])
    #fusion_layer = Dense(dim, activation = 'relu')(sum_layer)
    predicted =Dense(1, activation = 'sigmoid', name='predictor')(avg_layer)

    predictor = Model([input_data_m, input_data_r, input_data_mi, input_data_rp], predicted)

    #autoencoder.summary()#200w
    #predictor.summary()#100w
    #优化和损失函数
    predictor.compile(optimizer='SGD',
                        loss={ 'predictor': 'binary_crossentropy'},
                        loss_weights={ 'predictor': 1.})
    #训练
    predictor.fit([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']],
                     y_train,
                    epochs=epo,
                    batch_size=bs,
                    shuffle=True,
                    verbose=0)

    plot_model(predictor, to_file='M_NN_Sequential.png', show_shapes=True)

    train_roc=roc_auc_score(y_train, predictor.predict([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']]))
    test_roc=roc_auc_score(y_test, predictor.predict([X_test['methy'], X_test['rnaseq'], X_test['mirna'], X_test['rppa']]))

    print('result:', train_roc)
    print('result', test_roc)
    return test_roc

def M_NN_Sequential_MT(X_train, X_test, y_train, y_test, dim=100, epo=800, bs=1000):
    print('M_NN_Sequential_MTing...')
    #结构
    dim_max = max([X_train['methy'].shape[1], X_train['rnaseq'].shape[1], X_train['mirna'].shape[1], X_train['rppa'].shape[1]])

    input_data_m = Input(shape=(X_train['methy'].shape[1],))
    encoded_m = Dense(dim, activation = 'relu')(input_data_m)
    predicted_m =Dense(1, activation = 'sigmoid', name='predictor_methy')(encoded_m)

    input_data_r = Input(shape=(X_train['rnaseq'].shape[1],))
    encoded_r = Dense(dim, activation = 'relu')(input_data_r)
    predicted_r = Dense(1, activation = 'sigmoid', name='predictor_rnaseq')(encoded_r)

    input_data_mi = Input(shape=(X_train['mirna'].shape[1],))
    encoded_mi = Dense(dim, activation = 'relu')(input_data_mi)
    predicted_mi = Dense(1, activation = 'sigmoid', name='predictor_mirna')(encoded_mi)

    input_data_rp = Input(shape=(X_train['rppa'].shape[1],))
    encoded_rp = Dense(dim, activation = 'relu')(input_data_rp)
    predicted_rp =Dense(1, activation = 'sigmoid', name='predictor_rppa')(encoded_rp)

    avg_layer = layers.average([encoded_m, encoded_r, encoded_mi, encoded_rp])
    predicted =Dense(1, activation = 'sigmoid', name='predictor')(avg_layer)

    autoencoder = Model([input_data_m, input_data_r, input_data_mi, input_data_rp],
                        outputs=[predicted_m, predicted_r, predicted_mi, predicted_rp, predicted])
    #autoencoder = Model([input_data_m, input_data_r, input_data_mi, input_data_rp],
      #                  outputs=[decoded_m, decoded_r, predicted_m, predicted_r, predicted_mi, predicted_rp, predicted])
    predictor_m = Model(input_data_m, predicted_m)
    predictor_r = Model(input_data_r, predicted_r)
    predictor_mi = Model(input_data_mi, predicted_mi)
    predictor_rp = Model(input_data_rp, predicted_rp)
    predictor = Model([input_data_m, input_data_r, input_data_mi, input_data_rp], predicted)

    #autoencoder.summary()#200w
    #predictor.summary()#100w
    #优化和损失函数
    autoencoder.compile(optimizer='SGD',
                        loss={
                              'predictor_methy': 'binary_crossentropy',
                              'predictor_rnaseq': 'binary_crossentropy',
                              'predictor_mirna': 'binary_crossentropy',
                              'predictor_rppa': 'binary_crossentropy',
                              'predictor': 'binary_crossentropy'},
                        loss_weights={
                                      'predictor_methy': 1.,
                                      'predictor_rnaseq': 1.,
                                      'predictor_mirna': 1.,
                                      'predictor_rppa': 1.,
                                      'predictor': 1.})
    #训练
    autoencoder.fit([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']],
                    [ y_train, y_train, y_train, y_train, y_train],
                    epochs=epo,
                    batch_size=bs,
                    shuffle=True,
                    verbose=0)

    plot_model(autoencoder, to_file='M_NN_Sequential_MT.png', show_shapes=True)
    train_roc_methy=roc_auc_score(y_train, predictor_m.predict(X_train['methy']))
    test_roc_methy=roc_auc_score(y_test, predictor_m.predict(X_test['methy']))

    train_roc_rnaseq=roc_auc_score(y_train, predictor_r.predict(X_train['rnaseq']))
    test_roc_rnaseq=roc_auc_score(y_test, predictor_r.predict(X_test['rnaseq']))

    train_roc_mirna=roc_auc_score(y_train, predictor_mi.predict(X_train['mirna']))
    test_roc_mirna=roc_auc_score(y_test, predictor_mi.predict(X_test['mirna']))

    train_roc_rppa=roc_auc_score(y_train, predictor_rp.predict(X_train['rppa']))
    test_roc_rppa=roc_auc_score(y_test, predictor_rp.predict(X_test['rppa']))

    train_roc=roc_auc_score(y_train, predictor.predict([X_train['methy'], X_train['rnaseq'], X_train['mirna'], X_train['rppa']]))
    test_roc=roc_auc_score(y_test, predictor.predict([X_test['methy'], X_test['rnaseq'], X_test['mirna'], X_test['rppa']]))

    print('result_methy_train:', train_roc_methy)
    print('result_methy_test', test_roc_methy)
    print('result_rnaseq_train:', train_roc_rnaseq)
    print('result_rnaseq_test', test_roc_rnaseq)
    print('result_mirna_train:', train_roc_mirna)
    print('result_mirna_test', test_roc_mirna)
    print('result_rppa_train:', train_roc_rppa)
    print('result_rppa_test', test_roc_rppa)
    print('result:', train_roc)
    print('result', test_roc)
    return test_roc_methy, test_roc_rnaseq, test_roc_mirna, test_roc_rppa, test_roc

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def test_os(clfs):
    sample, target_os = load_data_os()
    for name, clf in clfs:
        print('clfing: ', name)
        result=cross_val_score(clf, sample, target_os, groups=target_os, scoring='roc_auc', cv=10, n_jobs=-1)
        #print(result)
        print('{}({})'.format(round(np.mean(result),4), round(np.std(result), 4)))
    return None

def test_dss(clfs):
    sample, target_dss = load_data_dss()
    for name, clf in clfs:
        print('clfing: ', name)
        result=cross_val_score(clf, sample, target_dss, groups=target_dss, scoring='roc_auc', cv=10, n_jobs=-1)
        #print(result)
        print('{}({})'.format(round(np.mean(result),4), round(np.std(result), 4)))
    return None

def test_pfi(clfs):
    sample, target_pfi = load_data_pfi()
    for name, clf in clfs:
        print('clfing: ', name)
        result=cross_val_score(clf, sample, target_pfi, groups=target_pfi, scoring='roc_auc', cv=10, n_jobs=-1)
        #print(result)
        print('{}({})'.format(round(np.mean(result),4), round(np.std(result), 4)))
    return None

def test_dfi(clfs):
    sample, target_dfi = load_data_dfi()
    for name, clf in clfs:
        print('clfing: ', name)
        result=cross_val_score(clf, sample, target_dfi, groups=target_dfi, scoring='roc_auc', cv=10, n_jobs=-1)
        #print(result)
        print('{}({})'.format(round(np.mean(result),4), round(np.std(result), 4)))
    return None



def test_dl_M():
    #datatypes = ['OS', 'DSS', 'PFI', 'DFI']
    datatypes = ['OS']
    for datatype in datatypes:
        print(datatype)
        sample, target = load_data(datatype)
        col_name = sample.columns
        num_methy = col_name.str.contains("methy").sum()
        num_rnaseq = col_name.str.contains("rnaseq").sum()
        num_mirna = col_name.str.contains("mirna").sum()
        num_rppa = col_name.str.contains("rppa").sum()

        sample = StandardScaler(1).fit_transform(sample)
        sample = pd.DataFrame(sample, columns=["methy"]*num_methy + ["rnaseq"]*num_rnaseq + ["mirna"]*num_mirna + ["rppa"] * num_rppa)

        cv=StratifiedKFold(n_splits=5)
        result=[[] for col in range(5)]
        for train_index, test_index in cv.split(sample, target):
            X_train, X_test, y_train, y_test = sample.iloc[train_index], sample.iloc[test_index], target[train_index], target[test_index]

            test_roc_methy, test_roc_rnaseq, test_roc_mirna, test_roc_rppa, test_roc=AE_T_M_F_MT(X_train, X_test, y_train, y_test, 400, epo=600, bs=1000)
            #test_roc=NN_Sequential(X_train, X_test, y_train, y_test, 100, epo=100, bs=1000)
            result[0].append(test_roc_methy)
            result[1].append(test_roc_rnaseq)
            result[2].append(test_roc_mirna)
            result[3].append(test_roc_rppa)
            result[4].append(test_roc)
        for i in range(5):
            print('{}({})'.format(round(np.mean(result[i]),4), round(np.std(result[i]), 4)))

    return None

def test_dl():
    # datatypes = ['OS', 'DSS', 'PFI', 'DFI']
    datatypes = ['OS']
    for datatype in datatypes:
        print(datatype)
        sample, target = load_data(datatype)

        col_name = sample.columns

        sample = StandardScaler(1).fit_transform(sample)
        sample = pd.DataFrame(sample, columns=col_name)

        cv=StratifiedKFold(n_splits=5)
        result=[]
        for train_index, test_index in cv.split(sample, target):
            X_train, X_test, y_train, y_test = sample.iloc[train_index], sample.iloc[test_index], target[train_index], target[test_index]

            # col_train, col_test = X_train.columns, X_test.columns
            #
            # scaler = StandardScaler().fit(X_train)
            # X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
            # X_train, X_test = pd.DataFrame(X_train, columns=col_train), pd.DataFrame(X_test, columns=col_test)
            # test_roc=M_NN_Sequential(X_train, X_test, y_train, y_test, 400, epo=100, bs=1000)
            test_roc=NN_Sequential(X_train, X_test, y_train, y_test, 100, epo=600, bs=1000)
            result.append(test_roc)
        print('{}({})'.format(round(np.mean(result),4), round(np.std(result), 4)))

    return None
print("start")
test_dl_M()
print("end")
