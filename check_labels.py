

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    "test",
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

model = load_model("emotion_model.h5")

loss, accuracy = model.evaluate(test_generator, verbose=1)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Get predictions for all test images
predictions = model.predict(test_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Display performance for each emotion
class_names = list(test_generator.class_indices.keys())

print("\nClassification Report:")
print(classification_report(
    true_classes,
    predicted_classes,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(true_classes, predicted_classes))


