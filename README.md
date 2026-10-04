🚘 CarValue | Used Car Price Prediction

CarValue is a machine learning project that provides insights into the used-car market and estimates the price of a car based on selected vehicle characteristics.

The project uses a Linear Regression model and an interactive Streamlit application that allows users to explore car listings, filter vehicles, and generate model-based price estimates.

🔗 Live Demo

Streamlit App:
(https://used-car-price-prediction-ml-project.streamlit.app/)

📌 Project Overview

The project combines a machine learning model with an interactive Streamlit application.

The application has three main sections:

Market Overview: Explore overall used-car market statistics and visualizations.
Browse Cars: Search and filter car listings based on different criteria.
Estimate a Price: Enter vehicle details and receive an estimated price from the trained Linear Regression model.

The application also provides a comparison between the predicted value and the median listing price for the selected brand.

✨ Features
📊 Market Overview

The Market Overview section provides:

Total number of cars in the dataset
Average asking price
Most frequently listed brand
Newest model year
Average price by brand
Average price by model year
Listings by fuel type
Minimum and maximum price range
Median listing price
🔎 Browse Cars

Users can explore the available car listings using:

Make or model search
Brand filter
Fuel type filter
Transmission filter
Model year range
Price range

The filtered results display:

Brand
Model
Year
Fuel type
Transmission
Engine capacity
Kilometers driven
Owner type
Price

Users can also download the filtered listings as a CSV file.

💰 Estimate a Price

The price estimation section allows users to enter:

Brand
Fuel type
Transmission
Engine size (CC)
Owner type
Safety rating

The application then uses the trained Linear Regression model to generate an estimated car value.

It also displays the median listing price for the selected brand for comparison.

Note: The predicted value is a model-based estimate and not a formal vehicle valuation. Actual car prices may vary depending on factors such as vehicle condition, location, service history, and other factors not represented in the training data.

🤖 Machine Learning
Model Used

Linear Regression

The trained model is stored as:

cars_model.pkl
Features Used by the Model

The model uses six input features:

Brand
Fuel_Type
Transmission
Engine_CC
Owner_Type
Safety_Rating

Categorical features are converted to numerical values before being passed to the trained model.

The application also verifies that the saved model expects the same feature order as the application.

📂 Dataset

The application uses the following dataset:

car_price_regression_data (1).csv

The dataset contains information related to used cars, including:

Brand
Model
Year
Fuel Type
Transmission
Engine Capacity
Kilometers Driven
Owner Type
Safety Rating
Price

The application uses the dataset not only for prediction inputs but also for market exploration, filtering, statistics, and visualizations.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Streamlit
Joblib
Jupyter Notebook
📁 Project Structure
Used-Car-Price-Prediction/
│
├── app.py
├── car_price_regression_data (1).csv
├── cars_model.pkl
├── README.md
└── requirements.txt
⚙️ How to Run Locally
1. Clone the repository
git clone YOUR_GITHUB_REPOSITORY_URL
2. Open the project folder
cd Used-Car-Price-Prediction
3. Install the required libraries
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

🧠 Application Workflow
Used Car Dataset
       ↓
Data Preprocessing
       ↓
Feature Selection
       ↓
Linear Regression Model
       ↓
Saved Model (cars_model.pkl)
       ↓
Streamlit Application
       ↓
 ┌───────────────┬────────────────┬──────────────────┐
 │ Market        │ Browse Cars    │ Estimate a Price │
 │ Overview      │                │                  │
 └───────────────┴────────────────┴──────────────────┘
🔮 Future Improvements

Possible improvements include:

Comparing Linear Regression with other regression algorithms
Improving model performance
Adding additional vehicle features
Adding more advanced model evaluation
Improving the prediction interface
Adding additional market analysis features
👨‍💻 Author

Shantanu Aptikar
