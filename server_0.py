from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

classs = {0: '0-800',1:'801-3200'}

import sys
sys.path.append('/home/kittikhun62/guts_video_visualization/static/models/model.h5')

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model = tf.keras.models.load_model('/home/kittikhun62/guts_video_visualization/static/models/model.h5')

# model1.make_predict_function()


# def predict_image1(img_path):
#     # Read the image and preprocess it
#     img = image.load_img(img_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = preprocess_input(x)
#     x = np.expand_dims(x, axis=0)
#     result = model1.predict(x)
#      return age[result.argmax()]

# def predict_image2(img_path):
#     # Read the image and preprocess it
#     img = image.load_img(img_path, target_size=(150, 150))
#     g = image.img_to_array(img)
#     g = preprocess_input(g)
#     g = np.expand_dims(g, axis=0)
#     result = model2.predict(g)
#     return gender[result.argmax()]

def predict_image(img_path):
    # Read the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model.predict(x)
    return classs[result.argmax()]


# # routes
# @app.route('/')
# def index():
#     return render_template('test_upload.html')

# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files['image']
#     image = Image.open(file)
#     # Process the image here
#     processed_image = image.rotate(0)
#     # Save the processed image to a file
#     processed_image.save('static/processed_image.png')
#     # Return the processed image file as a response
#     return render_template('upload_completed.html')
#     #return send_file('processed_image.png', mimetype='image/png')


@app.route('/')
def index():
    return render_template('test_upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Read the uploaded image and save it to a temporary file
        file = request.files['image']
        img_path = 'static/processed_image.png'
        file.save(img_path)
        # Predict the age
        BET_pred = predict_image(img_path)      

        # Render the prediction result
        return render_template('upload_completed.html', prediction=BET_pred)



if __name__ == '__main__':
    app.run(debug=True, port=8080)