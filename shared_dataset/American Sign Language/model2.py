import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from datetime import datetime
import matplotlib.pyplot as plt

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")

# Configuration
EPOCHS = 100
BATCH_SIZE = 256
INIT_LR = 1e-3
MIN_LR = 1e-6
DECAY_FACTOR = 0.5
PATIENCE = 10

def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch != 0:
        new_lr = lr * DECAY_FACTOR
        return max(new_lr, MIN_LR)
    return lr

def main():
    # GPU Setup
    log("Initializing GPU configuration")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            log(f"Configured {len(gpus)} GPUs")
        except RuntimeError as e:
            log(f"GPU error: {str(e)}", "ERROR")
    else:
        log("No GPUs found, using CPU", "WARNING")

    # Load data
    log("Loading preprocessed data")
    X_train = np.load("asl_landmark_X_train.npy")
    y_train = np.load("asl_landmark_y_train.npy")
    X_val = np.load("asl_landmark_X_val.npy")
    y_val = np.load("asl_landmark_y_val.npy")
    X_test = np.load("asl_landmark_X_test.npy")
    y_test = np.load("asl_landmark_y_test.npy")
    
    log(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Enhanced Model Architecture
    log("Building model architecture")
    model = Sequential([
        GaussianNoise(0.01, input_shape=(63,)),  # Data augmentation
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(29, activation='softmax')
    ])

    # Custom optimizer configuration
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=INIT_LR,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1),
        ModelCheckpoint('best_model.h5', save_best_only=True, verbose=1),
        LearningRateScheduler(lr_scheduler)
    ]

    # Training
    log("Starting model training")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )

    # Evaluation
    log("Evaluating model")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    log(f"Final Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    # Save final model
    model.save('final_model.h5')
    log("Saved final model")

    # Convert to TFLite with optimizations
    log("Converting to TFLite")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    with open('asl_model.tflite', 'wb') as f:
        f.write(tflite_model)
    log("TFLite model saved")

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plt.savefig('training_history.png')
    log("Saved training history plot")

if __name__ == "__main__":
    main()