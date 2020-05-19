from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template, jsonify
import json
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from gevent.pywsgi import WSGIServer

import sys
import os
import glob
global model,graph
import tensorflow as tf
graph = tf.get_default_graph()
app = Flask(__name__)

#MODEL_PATH = '.model_weights.hdf5'

#model = load_model('.model_weights.hdf5')



def model_predict(img_path,imangle,model):
    img = image.load_img(img_path, target_size=(75,75))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    inc_angle= np.array(imangle)
    preds = model.predict_classes(x,inc_angle)
    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('new 1.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        iangle=request.files['inc-angle']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        model = load_model('ship_iceberg.h5')

        preds = model_predict(file_path,iangle,model)
        ls=["iceberg","ship"]
        result = ls[preds[0]]             
        return result
    return None


if __name__ == '__main__':
      port = int(os.getenv('PORT', 8000))
     #app.run(host='0.0.0.0', port=port, debug=True)
      http_server = WSGIServer(('0.0.0.0', port), app)
      http_server.serve_forever()
     #app.run(debug=True)