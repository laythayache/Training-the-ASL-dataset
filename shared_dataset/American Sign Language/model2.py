import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Verify GPU availability
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# Load data
X_train = np.load("asl_landmark_X_train.npy")
y_train = np.load("asl_landmark_y_train.npy")
X_val = np.load("asl_landmark_X_val.npy")
y_val = np.load("asl_landmark_y_val.npy")
X_test = np.load("asl_landmark_X_test.npy")
y_test = np.load("asl_landmark_y_test.npy")

# Model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),
    Dense(29, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

# Train using GPU
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks
)

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('asl_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")