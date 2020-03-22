import os 
import re
import json
import requests

import numpy as np
# import tensorflow as tf
# from PIL import Image, ImageOps
# from werkzeug.utils import secure_filename
# from keras.preprocessing imposrt image
from flask import Flask, request, render_template, redirect, jsonify

from src.inference import NikeNet

app = Flask(__name__)

nike_net = NikeNet()

@app.route('/', methods=['GET'])
def get_index(): # query
    return render_template('index_new.html')

@app.route('/api', methods=['GET', 'POST'])
def shoe_detect():

    ''' Get File '''
    shoe_img = request.files['image'] # https://tech-blog.s-yoshiki.com/2018/07/210/
    ''' Save File to ./uploads '''
    # name = secure_filename(shoe_img.filename)
    img_path = os.path.join("./image/", shoe_img.filename)
    print("img_path: {}".format(img_path))
    shoe_img.save(img_path)

    ''' Find most similar with vector '''
    cand = nike_net.find_top10(img_path)
    resp = jsonify(cand)

    return resp



if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=9527)
