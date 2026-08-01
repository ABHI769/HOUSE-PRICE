# House Price Prediction App

This project is a Flask web application that predicts house prices using a machine learning model trained on synthetic Indian housing data. It also supports currency conversion for displaying prices in different currencies.

## Features

- Predict house prices based on property characteristics
- Web-based interface built with Flask and HTML/CSS/JavaScript
- Supports multiple currencies such as INR, USD, EUR, GBP, and more
- Includes a training script to generate data and retrain the model

## Project Structure

- app.py - Flask application and prediction API
- train_model.py - Generates synthetic housing data and trains the model
- currency.py - Currency conversion and formatting utilities
- data/housing_data.csv - Sample training data
- models/ - Trained model and metadata
- templates/ - HTML templates for the frontend
- static/ - CSS and JavaScript assets

## Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Training the Model

To generate the dataset and train the model:

```bash
python train_model.py
```

This will create:
- models/house_price_model.joblib
- models/model_metadata.json
- data/housing_data.csv

## Running the App

Start the Flask app:

```bash
python app.py
```

Then open your browser at:

```text
http://127.0.0.1:5000/
```

## API Usage

The app exposes a prediction endpoint:

- POST /api/predict

Example request body:

```json
{
  "area_sqft": 1200,
  "bedrooms": 3,
  "bathrooms": 2,
  "age_years": 5,
  "floor": 3,
  "total_floors": 10,
  "parking": 1,
  "city_tier": "metro",
  "property_type": "apartment",
  "furnishing": "semi",
  "currency": "USD"
}
```

## Notes

- The model is trained on synthetic data, so it is intended for demonstration and learning purposes.
- The app validates basic input checks such as positive area and logical floor values.

## Technologies Used

- Python
- Flask
- scikit-learn
- pandas
- numpy
- joblib
