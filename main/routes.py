"""Routes for parent Flask app."""
from flask import render_template, redirect, url_for
from flask import current_app as app
from transform_model import PATH_DIR
from extract import extract
from evaluate import test
from transform_model import transform_and_model
from recommend import recommend
import pandas as pd
import json
from flask_cors import CORS,  cross_origin


CORS(app)
@app.route('/')
def home():
    """Landing page."""
    #data = recommend()
    data = pd.read_csv(f'{PATH_DIR}/assets/recommendation.csv').drop(["index"], axis=1).to_dict(orient="index")


    return render_template(
        'index2.html',
        data = data

    )

@app.route('/my-link/')
@cross_origin() 
def my_link():
    print ('I got clicked!')    
    extract()
    transform_and_model()
    recommend()
    test()
    redirect(url_for('home'))

@app.route('/scheduler')
def process_run():
    extract()
    training, testing, high_testing, data = transform_and_model()



    """Landing page."""
    return render_template(
        'index2.html',
    )