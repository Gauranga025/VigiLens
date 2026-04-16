import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

data = np.load("training.npy")

model = Sequential()

model.add(Conv3D(128, (11,11,1), strides=(4,4,1),
                 activation='tanh', input_shape=(227,227,10,1)))

model.add(Conv3D(64, (5,5,1), strides=(2,2,1),
                 activation='tanh'))

model.add(ConvLSTM2D(64, (3,3), padding='same', return_sequences=True))
model.add(Dropout(0.3))

model.add(ConvLSTM2D(32, (3,3), padding='same', return_sequences=True))
model.add(Dropout(0.3))

model.add(ConvLSTM2D(64, (3,3), padding='same', return_sequences=True))

model.add(Conv3DTranspose(128, (5,5,1), strides=(2,2,1), activation='tanh'))
model.add(Conv3DTranspose(1, (11,11,1), strides=(4,4,1), activation='tanh'))

model.compile(optimizer='adam', loss='mse')

checkpoint = ModelCheckpoint("model/best_model.keras", save_best_only=True)
earlystop = EarlyStopping(patience=5)

model.fit(data, data,
          epochs=20,
          batch_size=4,
          validation_split=0.1,
          callbacks=[checkpoint, earlystop])

model.save("model/saved_model.keras")