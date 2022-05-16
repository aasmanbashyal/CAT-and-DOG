from flask import Flask ,request
import tensorflow as tf
import tensorflow_hub as hub
from keras.preprocessing import image
import numpy as np


app = Flask(__name__)

labels = ['CAT', 'DOG']

def load__model():
    model = tf.keras.models.load_model('model_catvsdog.h5', custom_objects={
                                       "KerasLayer": hub.KerasLayer})

    return model

model = load__model()

def image_preprocessing(img_path):
    img = image.load_img(img_path, target_size=(100, 100), color_mode='grayscale')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    X = np.vstack([x])

    return X/255.0

def predict(model, full_path):
    x = image_preprocessing(full_path)
    pred = model.predict(x)

    return pred

@app.route('/predict', methods = ['POST'])
def home():
    image_path = request.json['image_path']
    pred = predict(model, image_path)
    label = labels[np.argmax(pred)]
    prob = float(np.max(pred))

    return {"label" : label, "confidence" : prob}

if __name__ == '__main__':
    app.run(debug=True)


