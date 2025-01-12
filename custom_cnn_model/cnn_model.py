import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D, Flatten,
    BatchNormalization, LeakyReLU, Input, Add, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def load_and_preprocess_data(input_file, max_length=5000):
    df = pd.read_pickle(input_file)
    features = []
    for feature in df["features"]:
        if len(feature) > max_length:
            feature = feature[:max_length]
        elif len(feature) < max_length:
            feature = np.pad(feature, (0, max_length - len(feature)), mode='constant')
        features.append(feature)
    X = np.array(features)
    y = np.array(df["label"].tolist())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    X_train = X_train.reshape((X_train.shape[0], 50, -1, 1))
    X_test = X_test.reshape((X_test.shape[0], 50, -1, 1))

    return X_train, X_test, y_train, y_test, label_encoder

def create_advanced_deep_cnn(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # First block
    x = Conv2D(32, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)

    residual = Conv2D(64, kernel_size=(1, 1), padding='same')(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])  # Residual connection
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(128, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.4)(x)

    x = Conv2D(256, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.5)(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


if __name__ == "__main__":
    input_file = "labeled_features.pkl"
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(input_file, max_length=5000)

    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(label_encoder.classes_)
    model = create_advanced_deep_cnn(input_shape, num_classes)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    print("Training the model on GPU...")
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, reduce_lr]
    )

    print("Evaluating the model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    model.save("advanced_deep_cnn_model.h5")
    with open("label_encoder.pkl", "wb") as f:
        import pickle
        pickle.dump(label_encoder, f)

    print("Model and label encoder saved.")
