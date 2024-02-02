from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Load the model outside the route function to ensure it's loaded only once
model = tf.keras.models.load_model('predictor_resnet9_saved_model', compile=False)

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Normalize the image
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def diseasepred():
    # Main page
    return render_template('disease_pred.html',result='THE END')

@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        upload_dir = os.path.join(basepath, 'uploads')

        # Create 'uploads' directory if it doesn't exist
        if not os.path.exists(upload_dir):
            os.makedirs(upload_dir)

        file_path = os.path.join(upload_dir, secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(preds[0])

        disease_class = ['Cocci', 'Healthy', 'ncd', 'salmonella']
        ind = np.argmax(preds[0])
        print('Prediction:', disease_class[ind])
        result = disease_class[ind]
        print(result)

        # Return the result as JSON
        return render_template('disease_pred.html',result=result)

    return "Invalid Request Method"

if __name__ == "__main__":
    app.run(debug=True)
