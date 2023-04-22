from flask import Flask,request,jsonify
import os
import tensorflow as tf
from application.categories import categories
from PIL import Image as im
from application.grabcutSegmentation import GrabcutSegmentation
import numpy as np

UPLOAD_FOLDER = 'application/uploads'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return "Welcome to plant disease detection Application"

@app.route('/upload',methods=['POST'])
def upload():
    if 'file' not in request.files:
        print("hello")
        return "No File is Selected"    
    file = request.files['file']
    if file.filename == '':
        return "No File is Selected"
    if file and allowed_file(file.filename):
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        model = tf.keras.models.load_model('application/model/AISA_CNN.h5',compile=False)
        model.compile() #Paste it here

        path='application/uploads/'+filename
        original=tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
        image_path=GrabcutSegmentation.grabcut_segmentation(path)

        image_path = im.fromarray(image_path)
        image_path.save(path)
        new_img = tf.keras.preprocessing.image.load_img(path, target_size=(224,224))
        img = tf.keras.preprocessing.image.img_to_array(new_img)
        img = np.expand_dims(img, axis=0)
        img = img/255.0
        prediction = model.predict(img)
        print(path)
        os.remove(path)
        response= (categories[np.argmax(prediction)])
        return jsonify({"response":response})
    return "Not a Correct Format"






# sudo pip uninstall tensorflow
# sudo pip uninstall opencv-python
