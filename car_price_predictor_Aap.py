import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Set up Streamlit page configuration
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

# Title and description
st.title("🚗 Car Price Prediction App")
st.write("Welcome to the Car Price Predictor! This app helps estimate car prices based on several features.")

# Load dataset
data_path = 'car.csv'  # Update this path if needed
data = pd.read_csv(data_path)

# Display dataset preview and column names
st.write("### Dataset Preview")
st.write(data.head())
st.write("### Column Names in Dataset")
st.write(data.columns)

# Identify the target column (price) and confirm its name from the dataset
target_column = 'Selling_Price'  # Replace 'price' with the correct column name based on dataset columns displayed above

# Check if the target column exists
if target_column not in data.columns:
    st.error(f"The target column '{target_column}' was not found in the dataset. Please check the column names above and update the target column.")
else:
    # Separate features and target
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # One-hot encode categorical columns
    categorical_features = features.select_dtypes(include=['object']).columns
    features = pd.get_dummies(features, columns=categorical_features, drop_first=True)

    # Split data for model training and testing
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model evaluation
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    st.write("### Model Evaluation Metrics")
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")
    st.metric(label="R² Score", value=f"{r2:.2f}")

    # Sidebar for user inputs
    st.sidebar.header("Enter Car Features")
    user_input = {}
    for col in features.columns:
        if col in categorical_features:
            user_input[col] = st.sidebar.selectbox(f"{col}", options=features[col].unique())
        else:
            user_input[col] = st.sidebar.number_input(f"{col}", value=float(features[col].mean()))

    # Predict button
    if st.sidebar.button("Predict Price"):
        input_data = np.array([list(user_input.values())])
        predicted_price = model.predict(input_data)[0]
        st.sidebar.success(f"The estimated price of the car is: Rs{predicted_price:,.2f}lakh")

    # Feature importance visualization
    if hasattr(model, 'coef_'):
        st.write("### Feature Importance")
        feature_importance = pd.Series(model.coef_, index=features.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        
        # Plot feature importance
        fig, ax = plt.subplots()
        sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax, palette="viridis")
        ax.set_title("Feature Importance in Car Price Prediction")
        ax.set_xlabel("Coefficient Value")
        ax.set_ylabel("Features")
        st.pyplot(fig)

    # Additional styling
    st.sidebar.write("### About the App")
    st.sidebar.info(
        """
        This app uses a Linear Regression model trained on car data to predict car prices.
        Enter details of the car in the sidebar and click 'Predict Price' to get an estimate.
        """
    )