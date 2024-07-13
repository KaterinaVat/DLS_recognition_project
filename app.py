from flask import Flask, render_template, request
import h5py 
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import load_model
import numpy as np

app=Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    return render_template('index.html')




@app.route('/', methods=['POST'])
def predict():
    imagefile=request.files['imagefile']
    image_pass="./images/"+imagefile.filename
    imagefile.save(image_pass)

    image=load_img(image_pass, target_size=())
    image=img_to_array(image)
    image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    
    yhat=model.predict(image)
    label = yhat[0][0]

    classification = '%s (%.2f%%)' % (label[1], label[2]*100)

    return render_template('index.html', prediction=classification)


if __name__=='__main__':
    modelfile = 'model'
    print(".......Loading model.......")
    model = tf.keras.models.load_model('leaf_model.h5')
    print("Model has loaded")
    app.run(port=3000, debug=True)

