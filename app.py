import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import base64

# Add background image
def add_bg(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* All text black */
        h1, h2, h3, h4, h5, h6, p, label {{
            color: black !important;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg("bgg.jpg")

# Load data
df = pd.read_csv("IRIS.csv")

X = df.drop("species", axis=1)
y = df["species"]

# Train model
X_train, X_test, y_train, _ = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)

# Title
st.markdown(
    "<h1 style='text-align:center;'>Iris Flower Classifier</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Enter the measurements to predict the species.</p>",
    unsafe_allow_html=True
)

# Inputs
sepal_length = st.number_input("Sepal length", min_value=0.0)
sepal_width = st.number_input("Sepal width", min_value=0.0)
petal_length = st.number_input("Petal length", min_value=0.0)
petal_width = st.number_input("Petal width", min_value=0.0)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=X.columns
    )

    scaled = scaler.transform(input_data)
    prediction = model.predict(scaled)[0]

    st.markdown(
        f"""
        <div style='
            padding: 20px;
            margin-top: 20px;
            background: white;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 0 10px rgba(0,0,0,0.3);
        '>
            <h2 style='color: black;'>Predicted species: {prediction}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
