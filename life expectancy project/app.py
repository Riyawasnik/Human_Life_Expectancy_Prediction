#Importing Libraries
from flask import Flask, render_template, request
import pickle
import pandas as pd

#Creating Flask Application
app = Flask(__name__)

# Load the trained model
with open('LifeExpectancy_Model.sav', 'rb') as file:
    model = pickle.load(file)

#Rendering the Template
@app.route('/')
def index():
    return render_template('C://Users//riyaw//OneDrive//Desktop//life expectancy project//Templates//index.html', predict=None)

#Converting values to Float
@app.route('/', methods=["POST"])
def predict():
    data = request.form
    year = float(data['Year'])
    mortality = float(data['Mortality'])
    alcohol = float(data['Alcohol'])
    bmi = float(data['BMI'])
    polio = float(data['Polio'])
    diphtheria = float(data['Diphtheria'])
    gdp = float(data['GDP'])
    population = float(data['Population'])

#Organising the Input data    
    input_data = pd.DataFrame({
        'Year': [year],
        'Adult Mortality': [mortality],
        'Alcohol': [alcohol],
        'BMI': [bmi],
        'Polio': [polio],
        'Diphtheria': [diphtheria],
        'GDP': [gdp],
        'Population': [population]
    })

#Making Prediction for Input values
    prediction = model.predict(input_data)[0]
    
    return render_template("index.html", predict=prediction)

if __name__ == '__main__':
    app.run()









