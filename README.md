# A Deep Learning Predictor App

This is a Python program designed to perform predictions on football games using machine learning models. 
It consists of a Dash application that allows the user to visualize the results of the predictions and a script for training the model. The application retrieves the latest predictions from a CSV file, generates summary statistics, and displays them in a table. The training script reads data from a CSV file, preprocesses the data, trains the model, and saves the trained model and statistics to a JSON file.

# Dependencies
This program requires the following libraries such as : numpy, pandas, dash, plotly
install dependencies 
`pip install -r requirements.txt


The project contains the following files:

wsgi.py (the usual app.py):  application code that displays the predictions, statistics, and graphs and launch the dash app as one of the Flask routes
config.py : set the Flask default to be "wsgi.py" instead of "app.py"
transform_model.py: Script for training the machine learning model and saving the results to a JSON file
assets/recommendation.csv: training data (scraped)
assets/training_result.json: The training results fitted to the model in JSON format
results/: Directory containing the CSV files with the predictions. The latest file is used for displaying the predictions in the Dash app.
model/: Directory containing the saved model files.
main/routes.py: setting up the routes for the Flask components
main/: All Flask related components including static templates folders
main/__init__.py : Setting up as as the main Flask module but now can call into the Dash module
main/dash1 : the folder houses all the Dash component
README.md: This file


# Usage
Flask Dash App
To run the Dash application, simply run  
`python wsgi.py file`. 
The application will automatically retrieve the latest predictions from the CSV files and display them in a table along with summary statistics and graphs.

# Training and Testing Script
1. Complete one by one, in sequence
`python extract.py`
2. To train the machine learning model, run
`python train_model.py`
The script will read the extracted data files from one, preprocess the data, train the model, and save the results to a JSON file (assets/training_result.json)
3. To apply to new data for recommendation 
`python recommend.py`
4. To evaluate using test data, the last seven days
`python evaluate.py`

Acknowledgments
This program is inspired by the book "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron.