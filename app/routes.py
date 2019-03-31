from app import app
from flask import Flask, render_template, request, redirect, url_for, jsonify, session, escape, Response, flash, send_from_directory, Markup
from app.forms import CompleteForm

from werkzeug.utils import secure_filename
import os
from os import listdir
from os.path import isfile, join

from model1 import check, just_image_check, just_text_check

UPLOAD_FOLDER = os.path.basename('uploads')

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/favicon.ico')
def fav():
    print('Logo')
    return render_template('welcome.html')

@app.route('/complete', methods=['GET', 'POST'])
def complete():
    form = CompleteForm(request.form)
    if request.method == 'POST' and form.validate():
        news = {
            "headline": request.form['headline'],
            "body": request.form['body'],
        }
        print(news)
        file = request.files['image']
        file.save(os.path.join('./uploads', file.filename))
        # filename = secure_filename(form.image.data.filename)
        # form.image.data.save('uploads/' + filename)
        # f.save(os.path.join('uploads/' + filename))
        # return redirect(url_for('complete'))
        list_of_page_urls, credible_from_url_list, title_list, score = check(news['headline'],os.path.join('./uploads', file.filename))
        score_image = just_image_check(os.path.join('./uploads', file.filename))
        score_text = just_text_check(news['headline'], news['body'])
        if score == 1 and score_image == 1 and score_text == 1:
            result = "This is not a fake news."
        elif score == 0  and score_image == 0 and score_text == 0:
            result = "This is a fake news. Please read other relevent news sources before mentioning to others."
        elif [score, score_image, score_text].count(1) == 2:
            result = "We are pretty sure that this is not a fake news."
        elif [score, score_image, score_text].count(0) == 2:
            result = "We are pretty sure that this is a fake news. Please read other relevent news sources before mentioning to others."
        else:
            result = "This may be a fake news. Read related news carefully."
        #print(list_of_page_urls, credible_from_url_list, title_list, score)
        data = {
            "page_urls": list_of_page_urls,
            "crediable_from_url": credible_from_url_list,
            "titles": title_list,
            "score": score,
            "score_image": score_image,
            "score_text": score_text,
            "image_path": os.path.join('./uploads', file.filename),
            "result": result,
            "headline": news['headline'],
            "body": news['body']
        }

        print(data)
        # return redirect('/')
        return render_template('results_complete.html', data = data)
    return render_template('complete.html', form = form)
