import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model("predictor_resnet9_saved_model")

# Define a function to preprocess an image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image (same as during training)
    return img_array

# Specify the path to the image yo want to classify
image_path = "./validation/salmo/salmo.54.jpg"

# Preprocess the image
input_image = preprocess_image(image_path)

# Make predictions
predictions = model.predict(input_image)

# Assuming you have class labels defined during training
class_labels = ['Cocci','Healthy','ncd','salmonella']
"""
# Print the class labels and corresponding probabilities
for i in range(len(class_labels)):
    print(f"{class_labels[i]}: {predictions[0][i]}") """

# Assuming you have class labels defined during training, you can get the class with the highest probability
predicted_class_index = np.argmax(predictions)
predicted_class = class_labels[predicted_class_index]

print("Predicted Class:", predicted_class)

# Combine class labels and their corresponding probabilities
class_probabilities = [(class_labels[i], predictions[0][i]) for i in range(len(class_labels))]

# Sort the class_probabilities based on probabilities in descending order
class_probabilities.sort(key=lambda x: x[1], reverse=True)

# Print sorted class names and probabilities
for class_name, probability in class_probabilities:
    print(f"{class_name}: {probability:.8f}")

