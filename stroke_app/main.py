import streamlit as st
import pandas as pd
import numpy as np
import pickle


def assing_assing_to_cat(df, feature, feature_out, names, bins):
    """
    function to bin and categorise/rank (via names) numerical features
    """
    for name, bin0, bin1 in zip(names, bins[:-1], bins[1:]):
        df.loc[
            ((df[feature] > bin0) & (df[feature] <= bin1)), feature_out
        ] = name
        print(name)
    return df


st.header("Stroke Prediction App")

# load the pickle file
filename = "stroke_app/stroke_RF_model.pickle"
with open(filename, 'rb') as file:
	stroke_model = pickle.load(file)

st.subheader("Please enter the patient's properties:")


# gather input #####

# numerical
id_nr = st.number_input(
    "Insert the id", value=None, placeholder="Type a number..."
)

age = st.number_input(
    "Insert the age", value=None, placeholder="Type a number..."
)
bmi = st.number_input(
    "Insert the BMI", value=None, placeholder="Type a number..."
)
gluc = st.number_input(
    "Insert the avg. glucose level", value=None, placeholder="Type a number..."
)

# binary
gender = st.radio("Gender:", ["Male", "Female"])
ever_married = st.radio("Ever married:", ["Never married", "Married"])
Residence_type = st.radio("Residence type:", ["Rural", "Urban"])

# multiclass - handled by the preprocessor
work_type = st.radio(
    "Work type:",
    ["Never worked", "Children", "Govt job", "Private", "Self-employed"],
)
smoking_status = st.radio(
    "Smoking status:", ["Never smoked", "Unknown", "Formerly smoked", "Smokes"]
)
cardiovascular_health = st.radio(
    "Cardiovascular health:",
    ["Normal", "Hypertension (HT)", "Heart disease (HD)", "HD + HT"],
)


# create test data set for prediction:
column_order = [
    "id",
    "age",
    "avg_glucose_level",
    "bmi",
    "gender",
    "ever_married",
    "Residence_type",
    "work_type",
    "smoking_status",
    "cardiovascular_health",
]

data_order = [
    [
        id_nr,
        age,
        gluc,
        bmi,
        gender,
        ever_married,
        Residence_type,
        work_type,
        smoking_status,
        cardiovascular_health,
    ]
]

X_test = pd.DataFrame(data_order, columns=column_order)


# remap binary features to 1/0
binary_features = ["ever_married", "gender", "Residence_type"]
binary_map = [
    {"Married": 1, "Never married": 0},
    {"Female": 1, "Male": 0},
    {"Urban": 1, "Rural": 0},
]

for feature, pattern in zip(binary_features, binary_map):
    X_test[feature] = X_test[feature].map(pattern)


# bin the numerical features to resonable groups:
bins = [0, 20, 40, 60, 80, np.inf]
names = [0, 1, 2, 3, 4]
# ['Children', 'Young adult', 'Middle adult', 'Old adult', 'Very old adult']
X_test = assing_assing_to_cat(X_test, "age", "age_cat_r", names, bins)

bins = [0, 18.5, 25, 30, 35, 40, np.inf]
names = [1, 2, 3, 4, 5, 6]
# ['Underweight', 'Healthy weight', 'Overweight',
# 'Obesity 1', 'Obesity 2', 'Obesity 3']
X_test = assing_assing_to_cat(X_test, "bmi", "bmi_cat_r", names, bins)
X_test.loc[X_test["bmi"].isna(), "bmi_cat_r"] = 0  # BMI missing

A1C = [0.0, 5.7, 6.5, np.inf]
bins = 28.7 * np.asarray(A1C) - 46.7
names = [0, 1, 2]
# ['Normal', 'Prediabetes', 'Diabetes']
X_test = assing_assing_to_cat(
    X_test, "avg_glucose_level", "avg_glucose_level_cat_r", names, bins
)


if st.button("Make Prediction"):
    prediction = stroke_model.predict(X_test)

    print("\n ### final pred", np.squeeze(prediction, -1))

    if prediction == 1:
        prediction_answer = "a STROKE"
    else:
        prediction_answer = "NO stroke"

    st.markdown(f"**You are likey to have: {prediction_answer}**")

    st.write("Thank you! I hope you liked it.")
