import os
import sqlite3
from flask import Flask, request, session, g, redirect, url_for, abort, \
     render_template, flash

app = Flask(__name__) # create the application instance :)
app.config.from_object(__name__) # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.update(dict(
    SECRET_KEY='devkey',
    USERNAME='',
    PASSWORD=''
))
app.config.from_envvar('TIMELINE_SETTINGS', silent=True)

@app.route('/')
def main():
    return render_template('test.html')
