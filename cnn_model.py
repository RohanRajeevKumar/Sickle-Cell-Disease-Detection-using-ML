import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np

# Function to create CNN Model
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Function to train CNN Model
def train_cnn(features, labels, input_shape, num_classes, epochs=30, batch_size=32):
    # Convert features to numpy array & reshape if necessary
    X_train = np.array(features).reshape(-1, input_shape[0], input_shape[1], input_shape[2])
    y_train = tf.keras.utils.to_categorical(labels, num_classes)

    # Create model
    model = create_cnn_model(input_shape, num_classes)

    # Train model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Save Model
    model.save("trained_cnn_model.h5")
    print("CNN Model Trained & Saved Successfully!")
    return model
