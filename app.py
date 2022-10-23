
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from fastai.vision.all import *
from PIL import Image
from base64 import b64encode

import numpy as np
import pandas as pd

import time
import pickle
import os
import shutil
import math
import preprocess
import fastai

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/'
#app.config['trn_path'] = 'static/data'
#app.config['csv_path'] = 'static/labels_jpg.csv'

@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/display', methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        os.mkdir(app.config['UPLOAD_FOLDER'] + 'tmp')
        os.mkdir(app.config['UPLOAD_FOLDER'] + 'output')
        files = request.files.getlist('file')
        for file in files:
            filename = secure_filename(file.filename)
            file.save(app.config['UPLOAD_FOLDER'] + 'tmp/' + filename)
        preprocess.prepare_png('input', 'output', channels=(1, 2, 6), crop=False)
        shutil.rmtree(app.config['UPLOAD_FOLDER'] + 'tmp')
###
        #df = pd.read_csv(app.config['csv_path'])
        #labels = df[['ID', 'multilabel']]

        #tfms = L([Rotate(max_deg=15, p=0.5)])
        #dls = ImageDataLoaders.from_df(labels, app.config['trn_path'], label_delim='-', batch_tfms=tfms, seed = 42)
        scores = ['epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural']

        #learn = vision_learner(dls, densenet121, metrics=partial(accuracy_multi, thresh=0.5), model_dir = 'model')
        #learn.load('multilabel_densenet121_5')
        learn = load_learner('static/multilabel_densenet121_5.pkl')
        print('loaded')
        slices = os.listdir(app.config['UPLOAD_FOLDER'] + 'output')
        for slice in slices:
            image = Image.open(app.config['UPLOAD_FOLDER'] + 'output/' + slice)
            image = np.asarray(image)
            file_object = io.BytesIO()
            img = Image.fromarray(image.astype('uint8'))
            img.save(file_object, 'PNG')
            pic = ('data:image/png;base64,'+b64encode(file_object.getvalue()).decode('ascii'))
            preds = []
            diag, null, probs = learn.predict(app.config['UPLOAD_FOLDER'] + 'output/' + slice)
            print(diag)
            for i in scores:
                index = scores.index(i)
                preds.append(str(i) + ' ' + str(float(probs[index])*100)[:4] + '%')
                print(i, float(probs[index]))
        shutil.rmtree(app.config['UPLOAD_FOLDER'] + 'output')
###
    return render_template('content.html', preds = preds, pic = pic)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080, debug = False)
