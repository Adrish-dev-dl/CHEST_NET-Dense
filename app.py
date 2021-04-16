from __future__ import division, print_function
# coding=utf-8
import os
import cv2
import tensorflow as tf

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)


# Load your trained model
model = tf.keras.models.load_model('chest_net_deploy.h5')        # Necessary

print('Model loaded. Check http://127.0.0.1:5000/')


categories=['FIT_LUNGS','SICK_LUNGS']
def predict(filepath,model):
    img_size=110
    img_array=cv2.imread(filepath)
    img_array=cv2.resize(img_array,(img_size,img_size))
    new_array=img_array.reshape(-1,img_size,img_size,3) #[BATCHSIZE,*DIMENSIONS*,COLOR_CHANNELS]
    prediction=model.predict([new_array])
    if prediction[0][0] > 0.43:
        return categories[1]
    elif prediction[0][0] <= 0.43:
        return categories[0]


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = predict(file_path, model)

        return preds


if __name__ == '__main__':
    app.run(debug=True)
