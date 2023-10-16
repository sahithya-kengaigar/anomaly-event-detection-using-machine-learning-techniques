from keras.models import model_from_json # Parses a JSON model configuration string and returns a model instance.
from pathlib import Path
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2 # Capturing video using OpenCV
import os
from glob import glob
import pandas as pd 
import flask
#import pickle
from flask import Flask, render_template, request
import statistics, requests
app=Flask(__name__)

@app.route('/')
def home():

    #pickle.dump(model,open("model1.pkl","wb"))
    
    return flask.render_template('home.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    file = request.form['file']
    path = os.path.relpath(file)
    path = os.path.join('u_input',path)
    
    class_names = class_names = {
        0:'Arson',
        1:'Burglary',
        2:'Explosion',
        3:'Fighting',
        4:'Normal'
    }
    #model = pickle.load(open("model1.pkl","rb"))

    f = Path("models/model_structure.json")
    model_structure = f.read_text()
    model = model_from_json(model_structure) # Parses a JSON model configuration string and returns a model instance.

    
    # loading the trained weights
    model.load_weights("models/model_weights.h5")
    
    #a = os.listdir('u_input')
    
    # removing all other files from the temp folder
    files = glob('temp/*')
    for f in files:
        os.remove(f)
    predict = []
    count = 0
    cap = cv2.VideoCapture(path)   # capturing the video from the given path
    while(cap.isOpened()):
        # reading from frame 
        ret,frame = cap.read() 
        if ret:
            if count%300 == 0:
                filename = 'temp/'+"_frame%d.jpg" % count;
                # writing the extracted images 
                cv2.imwrite(filename, frame) 
            count+=1
        else: 
                break   
    cap.release() # Closes video file or capturing device.
    cv2.destroyAllWindows() # allows users to destroy all windows at any time.
    # reading all the frames from temp folder
    images = glob("temp/*.jpg")
    prediction_images = []
    for i in range(len(images)):
        img = image.load_img(images[i], target_size=(64,64,3)) #(height,width,rgb) #load image
        img = image.img_to_array(img) #Converts a PIL Image instance to a 3d-array
        img = img/255 #convert rgb(255) to 0-1 value
        prediction_images.append(img)
    # converting all the frames for a test video into numpy array
    prediction_images = np.array(prediction_images)
    y_pred = model.predict(prediction_images, batch_size=1, verbose=0)
    # prediction_images: Number of samples per batch of computation
    # verbose: By setting verbose 0, 1 or 2 you just say how do you want to 'see' the training progress for each epoch.
    # verbose=0 will show you nothing (silent)
    # verbose=1 will show you an animated progress bar.
    # verbose=2 will just mention the number of epoch.
    y_predict = []
    for i in range(0, len(y_pred)):
        y_predict.append(int(np.argmax(y_pred[i])))
    #return jsonify('This video clasify as a ',class_label)
    # The mode of a set of data values is the values that appears most often.
    def most_common(List):
            return statistics.mode(List)
    l = list(y_predict)
    # appending the mode of predictions in predict list to assign the tag to the video
    most_likely_class_index = most_common(l)
    result = class_names[most_likely_class_index]
    return render_template("home.html",result=result)

if __name__ == "__main__":
    app.run(debug=True)
