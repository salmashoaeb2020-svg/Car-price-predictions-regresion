# 🚗 Egyptian Car Price Prediction

A Machine Learning project that predicts used car prices in Egypt using Regression techniques.

## 📌 Project Overview

This project predicts the estimated price of a car in Egyptian Pounds (EGP) based on its specifications.

The user provides:

* Car Brand
* Car Model / Line
* Model Year
* Kilometers
* Car Condition
* Car Trim / Category
* Car Image

The machine learning model uses the car specifications to predict its estimated market price.

> The uploaded image is currently displayed in the application but is not used as an input feature by the regression model.

## 🧠 Machine Learning

The project uses:

* Python
* Pandas
* Scikit-learn
* Random Forest Regression
* One-Hot Encoding
* Train/Test Split

## 📊 Features

### Input Features

| Feature    | Type        |
| ---------- | ----------- |
| Brand      | Categorical |
| Model      | Categorical |
| Year       | Numerical   |
| Kilometers | Numerical   |
| Condition  | Categorical |
| Trim       | Categorical |

### Target

`Price`

The target is measured in Egyptian Pounds (EGP).

## 📁 Project Structure

```text
car_price_prediction/
│
├── data/
│   └── cars.csv
│
├── model/
│   ├── train_model.py
│   └── car_price_model.pkl
│
├── app/
│   └── app.py
│
├── notebooks/
│   └── analysis.ipynb
│
├── requirements.txt
│
└── README.md
```

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Go to the model directory:

```bash
cd model
```

Then:

```bash
python train_model.py
```

This will generate:

```text
car_price_model.pkl
```

### 3. Run the application

Go back to the project root:

```bash
cd ..
```

Then:

```bash
streamlit run app/app.py
```

The application will open in the browser.

## 🔮 Future Improvements

* Use a larger Egyptian car dataset.
* Add more car brands and models.
* Add transmission type.
* Add fuel type.
* Add engine capacity.
* Add city/location.
* Compare multiple regression algorithms.
* Improve model accuracy.
* Use Computer Vision to extract information from the uploaded car image.
* Detect the car brand/model automatically from the image.
* Deploy the application online.

