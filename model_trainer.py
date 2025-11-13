import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_model():
    """Create CNN model for brain tumor classification"""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dense(4, activation='softmax')  # 4 classes: glioma, meningioma, pituitary, no_tumor
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_model():
    """Train the brain tumor detection model using local dataset"""

    train_dir = './dataset/Training'  # path to training data
    test_dir = './dataset/Testing'    # path to testing data

    if not os.path.exists(train_dir):
        print(f"Training directory '{train_dir}' not found!")
        return

    if not os.path.exists(test_dir):
        print(f"Testing directory '{test_dir}' not found!")
        return

    # Data generators for training and testing
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    validation_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical'
    )

    model = create_model()

    print("Training the model...")
    history = model.fit(
        train_generator,
        epochs=10,  # increase epochs as needed for better accuracy
        validation_data=validation_generator
    )

    # Save model to disk
    model.save('brain_tumor_model.h5')
    print("Model trained and saved as 'brain_tumor_model.h5'.")

    print("Class indices:", train_generator.class_indices)

if __name__ == "__main__":
    train_model()
