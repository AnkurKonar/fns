import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash, send_from_directory
#from werkzeug import secure_filename
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/captainlazarus/projects/timeline/Timeline/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config.from_object(__name__) # load config from this file , flaskr.py
app.config.from_envvar('TIMELINE_SETTINGS', silent=True)

articles = ['1']
dates = ['13']

#Main Timeline
@app.route('/')
def hello_world():
    return render_template('test.html' , dates = dates , articles = articles)

#Utility Functions
def analyse_image(art , f):
    print(art , "\n" , f)

#Utility routes
@app.route('/upload', methods=['POST' , 'GET'])
def upload_file():
    if request.method == 'POST':
        art = request.form.to_dict()
        art = art['TextArea']
        print(art)
        file = request.files['image']
        f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

        # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
        file.save(f)

        analyse_image(art , f)

        return render_template('index1.html')
    else:
        return render_template('index1.html')