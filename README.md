A machine learning project that predicts the price of a used car based on selected vehicle features. The project includes data preprocessing, exploratory data analysis, feature engineering, model training, and a Streamlit web interface for making predictions.

Live URL: (https://used-car-price-prediction-ml-project.streamlit.app/)


Project Overview

The goal of this project is to build a simple machine learning model that can estimate the price of a used car based on information such as brand, fuel type, transmission, ownership type, and safety rating.

The project was developed using Python and Scikit-learn, with Streamlit used to create an interactive web application.

Dataset

The dataset contains information about used cars, including features such as:

Brand
Model
Year
Age
Fuel Type
Transmission
Engine Capacity
Kilometers Driven
Owner Type
Mileage
Seats
Safety Rating
Price
Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Streamlit
Jupyter Notebook
Machine Learning Model

Linear Regression was used to build the price prediction model.

The workflow includes:

Loading the dataset
Understanding the data
Data cleaning and preprocessing
Exploratory Data Analysis (EDA)
Feature engineering
Selecting relevant features
Splitting the data into training and testing sets
Training the Linear Regression model
Saving the trained model
Building the Streamlit application
Deploying the application
Features Used for Prediction

The deployed application uses selected vehicle features to generate a predicted price.

The selected features are:

Brand
Fuel Type
Transmission
Owner Type
Safety Rating
Project Structure
Used-Car-Price-Prediction/
│
├── app.py
├── cars_model.pkl
├── car_price_regression_data.csv
├── README.md
└── requirements.txt

File names may differ slightly depending on the final files in the repository.

How to Run Locally
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
2. Navigate to the project folder
cd Used-Car-Price-Prediction
3. Install the required libraries
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

Results

The trained Linear Regression model is integrated with the Streamlit application, allowing users to enter vehicle details and receive an estimated used-car price.

Future Improvements
Experiment with additional regression algorithms
Improve model performance through hyperparameter tuning
Add more relevant vehicle features
Improve the user interface
Compare different models and select the best-performing model
Author

Shantanu Aptikar
