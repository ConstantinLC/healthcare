import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from keras.models import Model
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

opt = optimizers.Adam(0.001)

def get_model(train):
    nclass = 1
    inp = Input(shape=(187, 1))
    
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid", trainable=train)(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid", trainable=train)(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1", trainable = True)(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2", trainable = True)(dense_1)
    dense_1 = Dense(5, activation=activations.sigmoid, name="dense_3_mitbih")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
    #model.summary()
    return model

model = get_model(True)
file_path = "./models/"+name+".h5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5, verbose=1)
redonplat = ReduceLROnPlateau(monitor="val_acc", mode="max", patience=3, verbose=2)
callbacks_list = [checkpoint, early, redonplat]  # early

model.load_weights('./models/baseline_cnn_mitbih.h5')
model.layers.pop()
model.layers[-1].outbound_nodes = []
model.outputs = [model.layers[-1].output]
output = Dense(1, activation = activations.sigmoid, name="added_dense")(model.layers[-1].output)
model_2 = Model(model.input, output)
model_2.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics=['acc'])
model_2.summary()

history = model_2.fit(X, Y, epochs=1000, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model_2.load_weights(file_path)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./accuracies/accuracy_'+name+'.png')
plt.clf()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('./losses/loss_'+name+'.png')

pred_test = model_2.predict(X_test)
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
