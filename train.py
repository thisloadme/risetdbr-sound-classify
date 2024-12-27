import json 
import os
import numpy as np
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import LSTM, Bidirectional, Dropout, Embedding, Activation
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from keras.optimizers import Adam
from keras.models import Model
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

current_directory = os.getcwd()
NUM_TAGS = 2
MFCC_MAX_LEN = 160
MFCC_NUM = 20
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 16
HIDDEN_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 0.01

config = []
with open(current_directory + '/mfcc_array_20.json', encoding='utf-8') as f:
    config = json.load(f)

mfcc_array = config['mfcc_array']
audio_target = config['audio_target']
# print(audio_target[:10])
# print(mfcc_array[:10])
# exit()

y = np.array(audio_target)

mfcc_array = np.array(mfcc_array)
X = mfcc_array
# X = np.interp(mfcc_array, (mfcc_array.min(), mfcc_array.max()), (0,1))

input_ = Input(shape=(MFCC_NUM,MFCC_MAX_LEN))
x = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True))(input_)
x = GlobalMaxPooling1D()(x)
x = Dense(64)(x)
x = Dropout(0.5)(x)
x = Activation('relu')(x)
x = Dense(32)(x)
# x = Dropout(0.5)(x)
x = Activation('relu')(x)
x = Dense(16)(x)
# x = Dropout(0.5)(x)
x = Activation('relu')(x)
output = Dense(NUM_TAGS, activation='sigmoid')(x)

model = Model(input_, output)
print(model.summary())
exit()
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)

# early_stopping = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)

r = model.fit(
    X,
    y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=VALIDATION_SPLIT
    # callbacks=[early_stopping]
)

# plt.plot(r.history['loss'], label='loss')
# plt.plot(r.history['val_loss'], label='val_loss')
# plt.legend()
# plt.show()

plt.plot(r.history['accuracy'], label='accuracy')
plt.plot(r.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

p = model.predict(X)
aucs = []
for j in range(NUM_TAGS):
    auc = roc_auc_score(y[:,j], p[:,j])
    aucs.append(auc)

print(np.mean(aucs))
# 0.6333060156367581

print(classification_report(np.argmax(y, axis=1), np.argmax(p, axis=1), target_names=np.array(['cat', 'dog'])))

conf_mat = confusion_matrix(np.argmax(y, axis=1), np.argmax(p, axis=1))
print(conf_mat)
fig, ax = plot_confusion_matrix(conf_mat=conf_mat,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True)
plt.show()
# exit()
model.save(current_directory + '/pre_trained_model.keras')