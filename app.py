# Importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

        
load_css()
# Setting up the app layout
# st.set_page_config(page_title="Medical Insurance Cost Prediction WebApp")
# st.markdown('<link rel="stylesheet" href="style.css">', unsafe_allow_html=True)
st.title("Medical Insurance Cost Prediction WebApp")

# Load the dataset
data = pd.read_csv("insurance.csv")

# Split the dataset into features (X) and target variable (y)
X = data.drop("charges", axis=1)
y = data["charges"]

# Perform one-hot encoding for categorical variables
cat_cols = ["sex", "smoker", "region"]
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)
st.set_option('deprecation.showPyplotGlobalUse', False)

# Age distribution
st.write("Knowledge about the dataset.")
st.subheader("Age Distribution")
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(data['age'])
plt.title('Age Distribution')
plt.show()
st.pyplot()

# Sex distribution
st.subheader("Sex Distribution")
sns.set()
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=data, palette="pastel")
plt.title('Sex Distribution')
st.pyplot()


# BMI distribution
st.subheader("BMI Distribution")
sns.set()
plt.figure(figsize=(6,6))
sns.histplot(data['bmi'])
plt.title('BMI Distribution')
plt.show()
st.pyplot()

# Children countplot
st.subheader("Children Distribution")
sns.set()
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=data)
plt.title('Children Distribution')
plt.show()
st.pyplot()

# Region countplot
st.subheader("Region Countplot")
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))
sns.countplot(x='region', data=data, palette="pastel")
plt.xlabel("Region")
plt.ylabel("Count")
plt.show()
st.pyplot()


# Charges distribution
st.subheader("Charges Distribution")
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(data['charges'])
plt.title("Charges distribution")
plt.show()
st.pyplot()

st.divider()

# Function to make predictions
def predict_charges(age, sex, bmi, children, smoker, region):
    # Perform one-hot encoding for the input variables
    input_data = pd.DataFrame(
        {
            "age": [age],
            "sex_male": [sex == "male"],
            "bmi": [bmi],
            "children": [children],
            "smoker_yes": [smoker == "yes"],
            "region_northwest": [region == "northwest"],
            "region_southeast": [region == "southeast"],
            "region_southwest": [region == "southwest"],
        }
    )
    # Reorder the columns to match the order in the training set
    input_data = input_data[X_train.columns]
    prediction = model.predict(input_data)
    return prediction[0]

def main():
    st.title("Predict your Insurance Charges")
    st.write("Enter the details to predict the insurance charges.")

    # Collect user input
    age = st.slider("Age", min_value=1, max_value=100, value=25, step=1)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.slider("BMI", min_value=10, max_value=50, value=25, step=1)
    children = st.slider("Children", min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox(
        "Region", ["southeast", "southwest", "northeast", "northwest"]
    )

    # Create a table to show the input values
    input_data = pd.DataFrame({
        "Input": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
        "Value": [age, sex, bmi, children, smoker, region]
    })
    st.table(input_data.set_index("Input"))

    # Make a prediction and display the result
    if st.button("Predict"):
        prediction = predict_charges(age, sex, bmi, children, smoker, region)
        st.write("The predicted insurance charges are:   $", prediction)

if __name__ == "__main__":
    main()

