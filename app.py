from flask import Flask, render_template, redirect, url_for
import os
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField
from wtforms.validators import DataRequired, Email, Length, EqualTo
from PIL import Image
import io
import cv2
import base64
from opencv_func import face_extractor
from facemodel import loadVggFaceModel, verify_face, preprocess_image
import json
from helper_func import write_json


app = Flask(__name__)

app.config['SECRET_KEY'] = 'mykey'

APP_ROOT = os.path.abspath(os.path.dirname(__file__))
REGISTERED_FACE_DIR = os.path.join(APP_ROOT, 'static/image/face_recognition/registered_faces/')
REQUESTED_FACE_DIR = os.path.join(APP_ROOT, 'static/image/face_recognition/requested_faces/')

users_json =  os.path.join(APP_ROOT, 'static/jsons/users.json')

model = loadVggFaceModel()

class ImageForm(FlaskForm):
    image = StringField('Image', validators = [DataRequired()])
    username = StringField('Username', validators = [DataRequired()])
    submit = SubmitField('Login')

class PasswordForm(FlaskForm):
    username = StringField('Username', validators = [DataRequired()])
    password = PasswordField('Password', validators = [DataRequired()])
    submit = SubmitField('Login')

class SignupForm(FlaskForm):
    fullname = StringField('Fullname', validators = [DataRequired()])
    username = StringField('Username', validators = [DataRequired()])
    email = StringField('Email', validators = [DataRequired()])
    password = PasswordField('Password', validators=[Length(min=5,max=20)])
    confirm_password = PasswordField('Confirm Password', validators=[Length(min=5,max=20),EqualTo('password',message='Password Must Match')])
    image = StringField('Image', validators = [DataRequired()])
    submit = SubmitField('Sign Up')

@app.route('/', methods = ["GET","POST"])
def index(message=''):
    form_1 = ImageForm()
    form_2 = PasswordForm()
    users_dict = open(users_json,)
    users_dict = json.load(users_dict)
    message = ''
    if form_1.validate_on_submit():
        username = form_1.username.data
        if username in users_dict.keys():
            base64String = form_1.image.data
            image = base64.b64decode(str(base64String))       
            fileName = username + '_test.jpeg'
            imagePath = REQUESTED_FACE_DIR + fileName
            img = Image.open(io.BytesIO(image))
            img.save(imagePath, 'jpeg')
            if face_extractor(imagePath) is not None:
                message = verify_face(username, face_extractor(imagePath), model, REGISTERED_FACE_DIR)
                return redirect(url_for('profile', username=username))
            else:
                message = 'No Face Found'
        else:
            message = 'User Not Registered'
    if form_2.validate_on_submit():
        username = form_2.username.data
        password = form_2.password.data
        if username in users_dict.keys():
            if users_dict[username][0]['password'] == password:
               return redirect(url_for('profile', username=username))
            else:
                message = "Incorrect Password"
        else:
            message = "User Not Registered"
    return render_template("webcamjs.html", form_1=form_1, form_2=form_2, message = message)


@app.route('/signup', methods = ['GET', 'POST'])
def signup():
    form = SignupForm()
    message = ''
    users_dict = open(users_json,)
    users_dict = json.load(users_dict)
    if form.validate_on_submit():
        username = form.username.data
        if username not in users_dict.keys():
            base64String = form.image.data
            fullname = form.fullname.data
            email = form.email.data
            password = form.password.data
            image = base64.b64decode(str(base64String))       
            fileName = username + '.jpeg'
            imagePath = REGISTERED_FACE_DIR + fileName
            img = Image.open(io.BytesIO(image))
            img.save(imagePath, 'jpeg')
            message = face_extractor(imagePath)
            if message is not None:
                new_user =  [{
                            "fullname": fullname,
                            "email": email,
                            "password": password,
                            }]
                
                write_json(new_user, users_json, username)
                message = 'Sign Up successful'
                return redirect(url_for('index', message = message))
            else:
                message = 'No Face Found'
        else:
            message = 'Username Already Registered'
    return render_template('signup.html', form = form, message = message)


@app.route('/<username>/profile', methods=["GET", "POST"])
def profile(username):
    users_dict = open(users_json,)
    users_dict = json.load(users_dict)
    fullname = users_dict[username][0]['fullname']
    email = users_dict[username][0]['email']
    return render_template('profile.html', fullname=fullname, email=email, username=username)

if __name__ == "__main__":
    app.run(debug=True)