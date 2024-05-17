from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from werkzeug.utils import secure_filename
import os
from wtforms.validators import InputRequired
import librosa
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder()

def extract_mfcc(y,sr):
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

@app.route('/', methods=['GET',"POST"])
@app.route('/home', methods=['GET',"POST"])
def home():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data # First grab the file
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'],secure_filename(file.filename))) # Then save the file
        songpath=os.path.join('C:\MainProject2.0\Flask','uploads',file.filename)
        y, sr = librosa.load(songpath, sr=None)
        x=extract_mfcc(y,sr)
        x=np.reshape(x,(1,-1))
        print(x)


        # Load the model from the pickle file
        with open('ser.pkl', 'rb') as file:
            model = pickle.load(file)

        print(x.shape)
        

        class_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Pleasnt surprise', 'sadness']

        results=model.predict(x)
        print(results)
        
        pred=class_labels[np.argmax(results[0])]
        print(pred)
        data={'result':pred}


        return render_template('results.html',data=data)
    return render_template('emotion.html', form=form)

if __name__ == '__main__':
    app.run()