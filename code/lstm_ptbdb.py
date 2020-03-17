import pandas as pd
import numpy as np

import tensorflow as tf
from keras import optimizers, losses, activations, models
# from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
# from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
#     concatenate, LSTM
# from keras.models import Sequential
# with tensorflow (because of this error :
# RuntimeError: Variable *= value not supported. Use `var.assign(var * value)` to modify the variable or `var = var * value` to get a new Tensor object.)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, LSTM, GRU, Masking, Bidirectional
from tensorflow.keras.models import Sequential
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import sys
name = sys.argv[0][:-3]
df_1 = pd.read_csv("../input/ptbdb_normal.csv", header=None)
df_2 = pd.read_csv("../input/ptbdb_abnormal.csv", header=None)
df = pd.concat([df_1, df_2])

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[187])


Y = np.array(df_train[187].values).astype(np.int8)
X = np.array(df_train[list(range(187))].values)[..., np.newaxis]

Y_test = np.array(df_test[187].values).astype(np.int8)
X_test = np.array(df_test[list(range(187))].values)[..., np.newaxis]


# TODO can try reduce the batch size (slower but might gives better score)
BATCH_SIZE = 512  # in stateful network number of samples need to be divisible by batch size (11641=7*1663 samples)
# train on 10476
# TODO could use two and only an argmax instead of what's done with pred_test ?
# nclass = 1
nclass = 2 # did not change anything from nclass=1 sigmoid, binary cross entr. test
SEED = 42


def build_model():
    # params
    dr = 0.2
    rec_dr = 0  # 0.2, dropping recurrent dropout made it predict more than majority class
    kern_reg = None  # tf.keras.regularizers.l2(l=0.001) None tf.keras.regularizers.l1(l=0.1)
    kern_init = tf.keras.initializers.glorot_normal()  # seed=SEED
    # create model
    model = Sequential()
    stateful = False  # (default false)
    model.add(Masking(mask_value=0., input_shape=(187, 1)))
    model.add(Bidirectional(GRU(128, dropout=dr, recurrent_dropout=rec_dr, return_sequences=True,
                   kernel_regularizer=kern_reg, kernel_initializer=kern_init)))
    # if stateful = true, did not work had dimension missmatch
    # model.add(LSTM(128, batch_input_shape=(BATCH_SIZE, 187, 1), dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True))
    # or could just add batch_size= with input_shape=
    # model.add(LSTM(128, input_shape=(187, 1), batch_size=BATCH_SIZE, dropout=0.2, recurrent_dropout=0.2, return_sequences=True, stateful=True))
    model.add(Bidirectional(GRU(64, dropout=dr, recurrent_dropout=rec_dr, return_sequences=True, stateful=stateful,
                   kernel_regularizer=kern_reg, kernel_initializer=kern_init)))
    model.add(Bidirectional(GRU(32, dropout=dr, recurrent_dropout=rec_dr, stateful=stateful,
                   kernel_regularizer=kern_reg, kernel_initializer=kern_init)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(nclass, activation='softmax'))  # TODO try sigmoid (when nclass=1?) softmax
    model.compile(loss='sparse_categorical_crossentropy',  # sparse_categorical_crossentropy, binary_crossentropy (nclass=1)
                  # todo is it equivalent to binary_crossentropy (try binary with sigmoid)?
                  # TODO maybe change to metrics=
                  optimizer='adam', weighted_metrics=['accuracy'])  # metrics=['accuracy'] weighted_metrics=['accuracy']

    model.summary()
    return model

model = build_model()
file_path = "./models/"+name+".h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

history = model.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy_'+name+'.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_'+name+'.png')

pred_test = model.predict(X_test)

precision, recall, thresholds = precision_recall_curve(Y_test, pred_test)
auc_prc = auc(recall, precision)
print("ROC_PRC score : %s"%auc_prc)
auc_roc = roc_auc_score(Y_test, pred_test)
print("Roc_AUC_score : %s"%auc_roc)

pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)




