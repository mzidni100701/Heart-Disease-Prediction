# Importing libraries
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from flask import Flask, request, jsonify, render_template
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load the trained model and scaler
loaded_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define the columns for the input DataFrame
columns = ['age', 'oldpeak', 'ca', 'sex_Male', 'dataset_Hungary',
           'dataset_Switzerland', 'dataset_VA Long Beach', 'cp_atypical angina',
           'cp_non-anginal', 'cp_typical angina', 'fbs_True', 'restecg_normal',
           'restecg_st-t abnormality', 'exang_True', 'slope_flat',
           'slope_upsloping', 'thal_normal', 'thal_reversable defect']

# Create a DataFrame with all values as 0 and columns as specified
df_template = pd.DataFrame(0, index=[0], columns=columns)

# Preprocessing function (similar to what you have in your code)
def preprocess_data(data):
    df_heart = pd.DataFrame(data, index=[0])  # Create a DataFrame from the input data

    select_num = ['age', 'oldpeak', 'ca']
    select_cat = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

    def fill_missing_values(df):
        # Mendapatkan kolom numerik dan kategorikal dengan missing values
        missing_numerical = df.select_dtypes(include=['number']).columns[df.select_dtypes(include=['number']).isnull().any()]
        missing_categorical = df.select_dtypes(include=['object']).columns[df.select_dtypes(include=['object']).isnull().any()]

        # Mengisi missing values untuk data numerik dengan modus
        for feature in missing_numerical:
            mode_value = df[feature].mode().iloc[0]
            df[feature].fillna(mode_value, inplace=True)

        # Mengisi missing values untuk data kategorikal dengan median
        for feature in missing_categorical:
            median_value = df[feature].median()
            df[feature].fillna(median_value, inplace=True)
        return df

    df_heart_filled = fill_missing_values(df_heart)

    def detect_and_handle_outliers(df_heart, features):
        class OutlierDetection:
            def __init__(self, features):
                self.features = features
                self.limit = {}

            def fit(self, df_heart):
                for feature in self.features:
                    self.limit[feature] = {}
                    percentile25 = df_heart[feature].quantile(0.25)
                    percentile75 = df_heart[feature].quantile(0.75)
                    iqr = percentile75 - percentile25
                    self.limit[feature]['upper'] = percentile75 + 1.5 * iqr
                    self.limit[feature]['lower'] = percentile25 - 1.5 * iqr

            def transform(self, df_heart):
                df_cap = df_heart.copy()
                for feature in self.features:
                    df_cap[feature] = np.where(
                        df_cap[feature] > self.limit[feature]['upper'],
                        self.limit[feature]['upper'],
                        np.where(
                            df_cap[feature] < self.limit[feature]['lower'],
                            self.limit[feature]['lower'],
                            df_cap[feature]
                        )
                    )
                return df_cap

        # Membuat dan menggunakan detektor outlier
        detector = OutlierDetection(features)
        detector.fit(df_heart)
        df_out = detector.transform(df_heart)
        return df_out

    features = ['age', 'oldpeak', 'ca']
    df_heart = detect_and_handle_outliers(df_heart, features)

    def ohe_data(df, categorical_cols, numerical_cols, scaler):
        # Memisahkan data kategorik dan numerik
        categorical_data = df[categorical_cols]
        numerical_data = df[numerical_cols]

        # Melakukan One-Hot Encoding pada data kategorik
        ohe = OneHotEncoder(sparse_output=False)
        categorical_encoded = ohe.fit_transform(categorical_data)

        # Membuat DataFrame dari hasil OHE
        categorical_encoded_df = pd.DataFrame(categorical_encoded, columns=ohe.get_feature_names_out(categorical_cols))

        # Melakukan Min-Max Scaling pada data numerik
        numerical_scaled = scaler.fit_transform(numerical_data)

        # Membuat DataFrame dari hasil scaling
        numerical_scaled_df = pd.DataFrame(numerical_scaled, columns=numerical_cols)

        # Menggabungkan kembali DataFrame yang telah diproses
        processed_df = pd.concat([numerical_scaled_df, categorical_encoded_df], axis=1)
        return processed_df

    df_heart = ohe_data(df_heart, numerical_cols=select_num, categorical_cols=select_cat, scaler=scaler)
    df_heart = df_heart.reindex(columns=df_template.columns, fill_value=0)
    return df_heart

# Prediction function
def predict_label(df_heart, loaded_model):
    predictions = loaded_model.predict(df_heart)
    label_mapping = {
        0: 'Tidak Ada',
        1: 'Kanker Stadium 1',
        2: 'Kanker Stadium 2',
        3: 'Kanker Stadium 3',
        4: 'Kanker Stadium 4'
    }
    return label_mapping.get(predictions[0], 'Unknown')

# Home route to render the form
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect data from the form
        data = {
            'age': request.form.get('age', type=float),
            'oldpeak': request.form.get('oldpeak', type=float),
            'ca': request.form.get('ca', type=int),
            'sex': request.form.get('sex'),
            'dataset': request.form.get('dataset'),
            'cp': request.form.get('cp'),
            'fbs': 'on' if request.form.get('fbs') == 'on' else 'off',
            'restecg': request.form.get('restecg'),
            'exang': 'on' if request.form.get('exang') == 'on' else 'off',
            'slope': request.form.get('slope'),
            'thal': request.form.get('thal')
        }

        # Preprocess the data and make a prediction
        processed_data = preprocess_data(data)
        prediction = predict_label(processed_data, loaded_model)

        # Render the template with the prediction result
        return render_template('index.html', prediction=prediction)

    except ValueError as ve:
        error_message = f"Input error: {str(ve)}. Please check the values and try again."
    except Exception as e:
        error_message = f"Unexpected error: {str(e)}."

    # If any error occurs, render the template with the error message
    return render_template('index.html', prediction=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))