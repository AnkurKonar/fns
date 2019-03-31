import os

class Config(object):
    # SECRET_KEY is required by flask-wtf module to avoid CSRF
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dont_guess'
    UPLOAD_FOLDER = os.path.basename('uploads')