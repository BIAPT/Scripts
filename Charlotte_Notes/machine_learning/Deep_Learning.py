import matplotlib
matplotlib.use('Qt5Agg')
from sklearn.model_selection import train_test_split
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import confusion_matrix
import seaborn as sn
from sklearn import metrics
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.callbacks import History
import keras.utils as utils
import random
from prepareDataset import *

X=X.iloc[:,empty==0]
X_Anes=X_Anes.iloc[:,empty==0]
X_Base=X_Base.iloc[:,empty==0]
X_Reco=X_Reco.iloc[:,empty==0]

random.seed(141)

X_train_Base,X_test_Base,Y_train_Base,Y_test_Base,Y_ID_Base_train,Y_ID_Base_test=train_test_split(X_Base,Y_out_Base,Y_ID_Base,random_state=0,test_size=0.3)
X_train_Reco,X_test_Reco,Y_train_Reco,Y_test_Reco,Y_ID_Reco_train,Y_ID_Reco_test=train_test_split(X_Reco,Y_out_Reco,Y_ID_Reco,random_state=0,test_size=0.3)
X_train_Anes,X_test_Anes,Y_train_Anes,Y_test_Anes,Y_ID_Anes_train,Y_ID_Anes_test=train_test_split(X_Anes,Y_out_Anes,Y_ID_Anes,random_state=0,test_size=0.3)

y_Anes_cat = utils.to_categorical(Y_out_Anes, 2)
y_Base_cat = utils.to_categorical(Y_out_Base, 2)
y_Reco_cat = utils.to_categorical(Y_out_Reco, 2)


hists_Anes=[]
hists_Base=[]
hists_Reco=[]

lrs=[0.0001,0.001,0.01,0.05,0.07,0.08]
len(lrs)

for l in lrs:
    model = Sequential()
    model.add(Dense(units=347, input_dim=347, activation='tanh'))
    model.add(Dense(units=100, activation='tanh'))
    model.add(Dense(2, activation='softmax'))

    sgd = optimizers.SGD(lr=l)
    model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])
    hist=model.fit(X_Anes, y_Anes_cat, validation_split=0.3,epochs=50, batch_size=4)
    hists_Anes.append(hist)
    hist=model.fit(X_Base, y_Base_cat, validation_split=0.3,epochs=50, batch_size=4)
    hists_Base.append(hist)
    hist=model.fit(X_Reco, y_Reco_cat, validation_split=0.3,epochs=50, batch_size=4)
    hists_Reco.append(hist)


plt.plot(hists_Base[0].history['loss'])
plt.plot(hists_Base[1].history['loss'])
plt.plot(hists_Base[2].history['loss'])
plt.plot(hists_Base[3].history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['0.0001','0.001','0.01','0.05'])
plt.title('Baseline')

plt.plot(hists_Reco[0].history['loss'])
plt.plot(hists_Reco[1].history['loss'])
plt.plot(hists_Reco[2].history['loss'])
plt.plot(hists_Reco[3].history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['0.0001','0.001','0.01','0.05'])
plt.title('Recovery')

plt.plot(hists_Anes[0].history['loss'])
plt.plot(hists_Anes[1].history['loss'])
plt.plot(hists_Anes[2].history['loss'])
plt.plot(hists_Anes[3].history['loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['0.0001','0.001','0.01','0.05'])
plt.title('Anesthesia')

plt.plot(hists_Anes[3].history['loss'])
plt.plot(hists_Anes[3].history['val_loss'])
plt.legend(['Training', 'Validation'], loc='upper left')
plt.title('Anesthesia learning_rate= 0.05')

plt.plot(hists_Base[3].history['loss'])
plt.plot(hists_Base[3].history['val_loss'])
plt.legend(['Training', 'Validation'], loc='upper left')
plt.title('Basline learning_rate= 0.05')

plt.plot(hists_Reco[2].history['loss'])
plt.plot(hists_Reco[2].history['val_loss'])
plt.legend(['Training', 'Validation'], loc='upper left')
plt.title('Recovery learning_rate= 0.01')



# FINAL MODEL Anes
model = Sequential()
model.add(Dense(units=347, input_dim=347, activation='tanh'))
model.add(Dense(units=347, activation='tanh'))
model.add(Dense(2, activation='softmax'))
sgd = optimizers.SGD(lr=0.05)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train_Anes, utils.to_categorical(Y_train_Anes, 2), validation_split=0, epochs=50, batch_size=4)
P2=model.predict_classes(X_test_Anes)
metrics.accuracy_score(Y_test_Anes, P2.astype(str))

# FINAL MODEL Base
model = Sequential()
model.add(Dense(units=347, input_dim=347, activation='tanh'))
model.add(Dense(units=347, activation='tanh'))
model.add(Dense(2, activation='softmax'))
sgd = optimizers.SGD(lr=0.05)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train_Base, utils.to_categorical(Y_train_Base, 2), validation_split=0, epochs=60, batch_size=4)
P2=model.predict_classes(X_test_Base)
metrics.accuracy_score(Y_test_Base, P2.astype(str))

# FINAL MODEL Reco
model = Sequential()
model.add(Dense(units=347, input_dim=347, activation='tanh'))
model.add(Dense(units=347, activation='tanh'))
model.add(Dense(2, activation='softmax'))
sgd = optimizers.SGD(lr=0.05)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.fit(X_train_Reco, utils.to_categorical(Y_train_Reco, 2), validation_split=0, epochs=40, batch_size=4)
P2=model.predict_classes(X_test_Reco)
metrics.accuracy_score(Y_test_Reco, P2.astype(str))


cv_DL_Base=[]
cv_DL_Anes=[]
cv_DL_Reco=[]


for r in range(0,4):
    for c in range (0,4):
        tmp_X_test_Base=X_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_X_train_Base=X_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]
        tmp_Y_test_Base=Y_out_Base[(Y_ID_Base == Part_reco[r]) | (Y_ID_Base == Part_chro[c])]
        tmp_Y_train_Base=Y_out_Base[(Y_ID_Base != Part_reco[r]) & (Y_ID_Base != Part_chro[c])]

        tmp_X_test_Anes=X_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_X_train_Anes=X_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]
        tmp_Y_test_Anes=Y_out_Anes[(Y_ID_Anes == Part_reco[r]) | (Y_ID_Anes == Part_chro[c])]
        tmp_Y_train_Anes=Y_out_Anes[(Y_ID_Anes != Part_reco[r]) & (Y_ID_Anes != Part_chro[c])]

        tmp_X_test_Reco=X_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_X_train_Reco=X_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]
        tmp_Y_test_Reco=Y_out_Reco[(Y_ID_Reco == Part_reco[r]) | (Y_ID_Reco == Part_chro[c])]
        tmp_Y_train_Reco=Y_out_Reco[(Y_ID_Reco != Part_reco[r]) & (Y_ID_Reco != Part_chro[c])]

        tmp_Y_train_Anes = utils.to_categorical(tmp_Y_train_Anes, 2)
        tmp_Y_train_Base = utils.to_categorical(tmp_Y_train_Base, 2)
        tmp_Y_train_Reco = utils.to_categorical(tmp_Y_train_Reco, 2)

        # FINAL MODEL Anes
        model = Sequential()
        model.add(Dense(units=347, input_dim=347, activation='tanh'))
        model.add(Dense(units=347, activation='tanh'))
        model.add(Dense(2, activation='softmax'))

        sgd = optimizers.SGD(lr=0.05)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(tmp_X_train_Anes, tmp_Y_train_Anes, validation_split=0, epochs=40, batch_size=4)
        P2 = model.predict_classes(tmp_X_test_Anes)
        accura=metrics.accuracy_score(tmp_Y_test_Anes, P2.astype(str))
        cv_DL_Anes.append(accura)

        # FINAL MODEL Base
        model = Sequential()
        model.add(Dense(units=347, input_dim=347, activation='tanh'))
        model.add(Dense(units=347, activation='tanh'))
        model.add(Dense(2, activation='softmax'))

        sgd = optimizers.SGD(lr=0.05)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(tmp_X_train_Base, tmp_Y_train_Base, validation_split=0, epochs=60, batch_size=4)
        P2 = model.predict_classes(tmp_X_test_Base)
        accura=metrics.accuracy_score(tmp_Y_test_Base, P2.astype(str))
        cv_DL_Base.append(accura)

        # FINAL MODEL Reco
        model = Sequential()
        model.add(Dense(units=347, input_dim=347, activation='tanh'))
        model.add(Dense(units=347, activation='tanh'))
        model.add(Dense(2, activation='softmax'))

        sgd = optimizers.SGD(lr=0.05)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        model.fit(tmp_X_train_Reco, tmp_Y_train_Reco, validation_split=0, epochs=40, batch_size=4)
        P2 = model.predict_classes(tmp_X_test_Reco)
        accura=metrics.accuracy_score(tmp_Y_test_Reco, P2.astype(str))
        cv_DL_Reco.append(accura)


plt.plot(cv_DL_Base)
plt.plot(cv_DL_Anes)
plt.plot(cv_DL_Reco)
plt.title('Neural Network')
plt.xlabel('cross-validation')
plt.ylabel('Accuracy')
plt.legend(['Baseline','Anesthesia','Recovery'])

np.mean(cv_DL_Base)
np.std(cv_DL_Base)
np.mean(cv_DL_Anes)
np.std(cv_DL_Anes)
np.mean(cv_DL_Reco)
np.std(cv_DL_Reco)
