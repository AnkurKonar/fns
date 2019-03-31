import os
import io
import sys
from bs4 import BeautifulSoup
import requests
import array as arr
import six as six
import gensim

from google.cloud import vision
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from PIL import Image
import os
from pylab import *
import re
from PIL import Image, ImageChops, ImageEnhance

import pandas as pd
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
# from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model

import pickle
credential_path = './key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

model1 = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin.gz', binary=True, limit = 500000)
credible = ['economictimes.', 'huffingtonpost.', 'theprint.', 'thelogicalindian.', 'thequint.', 'altnews.', 'wsj.', 'nypost.', 'nytimes.', 'bbc.', 'reuters.', 'economist.', 'pbs.', 'aljazeera.', 'thewire.', 'theatlantic.', 'theguardian.', 'edition.cnn','cnbc.', 'scroll.in', 'financialexpress.', 'npr.', 'usatoday.', 'snopes.', 'politifact.']

def entity_sentiment_text(text):
    """Detects entity sentiment in the provided text."""
    client = language.LanguageServiceClient()

    if isinstance(text, six.binary_type):
        text = text.decode('utf-8')

    document = types.Document(
        content=text.encode('utf-8'),
        type=enums.Document.Type.PLAIN_TEXT)

    # Detect and send native Python encoding to receive correct word offsets.
    encoding = enums.EncodingType.UTF32
    if sys.maxunicode == 65535:
        encoding = enums.EncodingType.UTF16

    result = client.analyze_entity_sentiment(document, encoding)

    #for entity in result.entities:
        #print('Mentions: ')
        #print(u'Name: "{}"'.format(entity.name))
        #for mention in entity.mentions:
            #print(u'  Begin Offset : {}'.format(mention.text.begin_offset))
            #print(u'  Content : {}'.format(mention.text.content))
            #print(u'  Magnitude : {}'.format(mention.sentiment.magnitude))
            #print(u'  Sentiment : {}'.format(mention.sentiment.score))
            #print(u'  Type : {}'.format(mention.type))
        #print(u'Salience: {}'.format(entity.salience))
        #print(u'Sentiment: {}\n'.format(entity.sentiment))

#---------------------#--------------------#---------------------#--------------------#
#Function for google's clous vision API
def detect_web(path):
    list = []
    i = 0
    """Detects web annotations given an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    # if annotations.best_guess_labels:
    #     for label in annotations.best_guess_labels:
    #         print('\nBest guess for the image: {}'.format(label.label))
    #         print("--------------------------------------------------------------------------------")


    if annotations.pages_with_matching_images:
        # print('\n{} Pages with matching images found:'.format(
        #     len(annotations.pages_with_matching_images)))

        for page in annotations.pages_with_matching_images:
            # print('\n\tPage url   : {}'.format(page.url))
            list.append(page.url)

    # if annotations.web_entities:
    #     print('\n{} Web entities found in the image: '.format(
    #         len(annotations.web_entities)))

    #     for entity in annotations.web_entities:
    #         print('\n\tScore      : {}'.format(entity.score))
    #         print(u'\tDescription: {}'.format(entity.description))

    # if annotations.visually_similar_images:
    #     print('\n{} visually similar images found:\n'.format(
    #         len(annotations.visually_similar_images)))

    #     for image in annotations.visually_similar_images:
    #         print('\tImage url    : {}'.format(image.url))
    # print("--------------------------------------------------------------------------------")
    return(list)
#---------------------#--------------------#---------------------#--------------------#
#Function to check which URLs belong to credible news sources
def credible_list(list_of_page_urls):

    c_list = []

    c_length = len(credible)

    url_length = len(list_of_page_urls)

    f = [[0 for j in range(c_length)] for i in range(url_length)]
    for i in range(url_length):
        for j in range(c_length):
            f[i][j] = list_of_page_urls[i].find(credible[j])
            if((list_of_page_urls[i].find(credible[j])) > 0):
                c_list.append(list_of_page_urls[i])
    if c_list == []:
        print("No credible sources have used this image, please perform human verification.")
        print("--------------------------------------------------------------------------------")
        exit(1)
    return(c_list)
#---------------------#--------------------#---------------------#--------------------#
#Function to scrape titles off the given URLs
def titles(credible_from_url_list):
    title_list = []
    for urls in credible_from_url_list:
        if urls != []:
            r = requests.get(urls)
            html = r.content
            soup = BeautifulSoup(html, 'html.parser')
            title_list.append(soup.title.string)
    return(title_list)

#---------------------#--------------------#---------------------#--------------------#
# #Function to print the scraped titles
# def print_article_title(title_list):
#     print("Credible article titles which use the same image: ")
#     print("--------------------------------------------------------------------------------")
#     for title in title_list:
#         print(title)
#         print("--------------------------------------------------------------------------------")
#---------------------#--------------------#---------------------#--------------------#
#Function to call google's language API for entity analysis
def entity_analysis(title_list):
    for title in title_list:
        entity_sentiment_text(title)

#---------------------#--------------------#---------------------#--------------------#
#Function to compute the WM distances between titles and associated title and the average distance
def wmdist(given_title,title_list):
    # print("Word Mover's Distance for Titles:")
    # print("--------------------------------------------------------------------------------")
    distances = []
    for title in title_list:
        dist = model1.wmdistance(given_title, title) #determining WM distance
        distances.append(dist)
        #distance = model1.WmdSimilarity(given_title, title)

    sum_dist = 0
    for distance in distances:
        sum_dist = sum_dist + distance
        # print ('distance = %.3f' % distance)
        # print("--------------------------------------------------------------------------------")

    avg_dist = sum_dist/len(distances)
    # print("Average Distance: {}".format(avg_dist))
    # print("--------------------------------------------------------------------------------")
    return(avg_dist)

#---------------------#--------------------#---------------------#--------------------#
#Function to decide whether human verification is required
def human_ver(avg_dist):
    if(avg_dist >= 1.0):
        return 0
    else:
        return 1

def check(given_title,image_path):
    list_of_page_urls = []
    credible_from_url_list = []
    title_list = []
    list_of_page_urls = detect_web(image_path)
    credible_from_url_list = credible_list(list_of_page_urls)
    title_list = titles(credible_from_url_list)
    entity_analysis(title_list)
    avg_dist = wmdist(given_title, title_list)
    score = human_ver(avg_dist)
    return list_of_page_urls, credible_from_url_list, title_list, score

#-----------------

def get_imlist(path):
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

def convert_to_ela_image(path, quality):
    filename = path
    resaved_filename = filename.split('.')[0] + '.resaved.jpg'
    ELA_filename = filename.split('.')[0] + '.ela.png'
    
    im = Image.open(filename).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    
    ela_im = ImageChops.difference(im, resaved_im)
    
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im

def just_image_check(image_path):
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', activation ='relu', input_shape = (128,128,3)))
    print("Input: ", model.input_shape)
    print("Output: ", model.output_shape)

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'valid', activation ='relu'))
    print("Input: ", model.input_shape)
    print("Output: ", model.output_shape)

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.25))
    print("Input: ", model.input_shape)
    print("Output: ", model.output_shape)

    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = "softmax"))

    optimizer = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

    model.load_weights('./model_image-05-0.9427.hdf5')
    x_test_arr = []
    x_test = image_path
    x_test_arr.append(array(convert_to_ela_image(x_test, 50).resize((128, 128))).flatten() / 255.0)
    x_test_arr = np.array(x_test_arr)
    x_test_arr = x_test_arr.reshape(-1, 128, 128, 3)
    arr = model.predict_classes(x_test_arr)
    return arr[0]

from nltk.tokenize import sent_tokenize, word_tokenize 
import nltk
nltk.download('punkt')
import csv
import re
from nltk.tokenize import WordPunctTokenizer
from bs4 import BeautifulSoup
import nltk
from autocorrect import spell
import itertools

tok = WordPunctTokenizer()
pat1 = r'@[A-Za-z0-9_]+'
pat2 = r'https?://[^ ]+'
combined_pat = r'|'.join((pat1, pat2))
www_pat = r'www.[^ ]+'
contractions = {
    "ain't": "am not / are not / is not / has not / have not",
    "aren't": "are not / am not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he'll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he's": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is / how does",
    "I'd": "I had / I would",
    "I'd've": "I would have",
    "I'll": "I shall / I will",
    "I'll've": "I shall have / I will have",
    "I'm": "I am",
    "I've": "I have",
    "isn't": "is not",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it's": "it has / it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what're": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who's": "who has / who is",
    "who've": "who have",
    "why's": "why has / why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "you'd've": "you would have",
    "you'll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you're": "you are",
    "you've": "you have"
}
contractions_pattern = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')



def text_cleaner(text):
    myre = re.compile(u'ud[a-z0-9]{3}', re.UNICODE)
    text = myre.sub(r' ', text)
    souped = BeautifulSoup(text, 'lxml').get_text()
    try:
        clean = souped.decode('utf-8-sig').replace(u'\ufffd','?')
    except:
        clean = souped
    stripped = re.sub(combined_pat, '', clean)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()
    neg_handled = contractions_pattern.sub(lambda x: contractions[x.group()], lower_case)
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]
    words = [spell(x) for x in words]
    return (" ".join(words)).strip()

def just_text_check(headline, text):
    news = text_cleaner(headline) + ' \n' + text_cleaner(text)
    test = open("test.txt", "w")
    test.write(news)
    test.close()
    test_arr = ['test.txt']
    vectorizer = TfidfVectorizer(input='filename', max_features=80, stop_words='english',encoding='utf-8',decode_error ='ignore',use_idf=True)
    news_vec = vectorizer.fit_transform(test_arr)
    with open('./logistic.pkl', 'rb') as f:
        logr = pickle.load(f)
    result = logr.predict(news_vec)
    if result[0] == True:
        return 1
    else:
        return 0
        