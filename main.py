
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from fastai.vision.all import *
from PIL import Image
from base64 import b64encode
from google.cloud import storage

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

app.config['UPLOAD_FOLDER'] = '/tmp'
#app.config['UPLOAD_FOLDER'] = 'static/'
#app.config['trn_path'] = 'static/data'
#app.config['csv_path'] = 'static/labels_jpg.csv'

# Instantiates a client
#storage_client = storage.Client()
# The name for the new bucket
#bucket_name = "richard-alpha-bucket"
# Creates the new bucket
#bucket = storage_client.create_bucket(bucket_name)
#bucket = storage_client.bucket(bucket_name)
#print(f"Bucket {bucket.name} created.")

@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/display', methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        dirpath = os.path.join('static', 'tmp')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        dirpath = os.path.join('static', 'output')
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        os.mkdir('static/tmp')
        os.mkdir('static/output')
        files = request.files.getlist('file')
        for file in files:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            storage_client = storage.Client()
            bucket_name = "richard-alpha-bucket"
            bucket = storage_client.get_bucket(bucket_name)
            #blob = bucket.blob(filename)
            blob = storage.Blob(filename, bucket)
            blob.upload_from_filename(filename)
            blob.download_to_filename('static/tmp/' + filename)
            #file.save(app.config['UPLOAD_FOLDER'] + 'tmp/' + filename)
            blob.delete()
        preprocess.prepare_png('input', 'output', channels=(1, 2, 6), crop=False)
        print('preprocessed')
        shutil.rmtree('static/tmp')
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
        slices = os.listdir('static/output')
        predictions = []
        images = []
        for slice in slices:
            image = Image.open('static/output/' + slice)
            image = np.asarray(image)
            file_object = io.BytesIO()
            img = Image.fromarray(image.astype('uint8'))
            img.save(file_object, 'PNG')
            #pic = ('data:image/png;base64,'+b64encode(file_object.getvalue()).decode('ascii'))
            images.append('data:image/png;base64,'+b64encode(file_object.getvalue()).decode('ascii'))
            print('saved')
            preds = []
            diag, null, probs = learn.predict('static/output/' + slice)
            print(diag)
            for i in scores:
                index = scores.index(i)
                preds.append(str(i) + ' ' + str(float(probs[index])*100)[:4] + '%')
                print(i, float(probs[index]))
            predictions.append(preds)
        shutil.rmtree('static/output')
###
    return render_template('content.html', predictions = predictions, images = images)

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080, debug = False)
