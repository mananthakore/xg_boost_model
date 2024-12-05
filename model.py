import dash
import base64
import io
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import plotly.express as px

# Initialize app
app = dash.Dash(__name__)

# Global variables
uploaded_data = None
model = None

# App layout
app.layout = html.Div([
    html.H1("Regression Prediction App"),

    # Upload Component
    html.Div([
        html.H2("1. Upload Dataset"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV File'),
            multiple=False
        ),
        html.Div(id='upload-feedback')
    ], style={'margin-bottom': '20px'}),

    # Select Target Component
    html.Div([ #this is where target variables are selected.
        html.H2("2. Select Target Variable"),
        dcc.Dropdown(id='target-dropdown', placeholder='Select the target variable'),
        html.Div(id='target-feedback')
    ], style={'margin-bottom': '20px'}),

    # Bar Charts Component
    html.Div([
        html.H2("3. Analyze Data"),
        html.Div([
            dcc.RadioItems(id='categorical-radio', inline=True),
            dcc.Graph(id='category-barchart')
        ]),
        dcc.Graph(id='correlation-barchart')
    ], style={'margin-bottom': '20px'}),

    # Train Component
    html.Div([
        html.H2("4. Train Model"),
        html.Div([dcc.Checklist(
            id='features-checkboxes')]),
        html.Button('Train Model', id='train-button'),
        html.Div(id='model-feedback')
    ], style={'margin-bottom': '20px'}),

    # Predict Component
    html.Div([
        html.H2("5. Make Predictions"),
        dcc.Input(id='predict-input', placeholder='Enter feature values...', style={'width': '80%'}),
        html.Button('Predict', id='predict-button'),
        html.Div(id='prediction-output')
    ])
])

# _______________helper functions_____________________________
# def preprocessing(uploaded_da):
#     df = uploaded_da
#     print(df)
#     # #did this for data cleansing so that we can handle missing data better.
#     ##this doesn't do anything because we know that our data does nto have any null values in the numerical columns, but just for good practice.
#     numeric_columns = df.select_dtypes(include=[np.number]).columns
#     df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
#
#     # use one hot encoding so that the machine learning model so that all values are treated equally.
#     encoder = OneHotEncoder(drop='first', sparse_output=False)
#     categorical_columns = df.select_dtypes(include=['object', 'category']).columns
#     encoded_df= encoder.fit_transform(categorical_columns)
#     encoded_df_columns = encoder.get_feature_names_out(categorical_columns)
#
#
#     # put the encoded features back into the data set.
#     df_encoded = pd.concat(
#         [df.drop(columns=categorical_columns),
#          pd.DataFrame(encoded_df, columns=encoded_df_columns)],
#         axis=1
#     )
#
#     # got rid out outliers so that predictions could be more accurate.
#     Q1 = df_encoded.quantile(0.25)
#     Q3 = df_encoded.quantile(0.75)
#     IQR = Q3 - Q1
#     df_encoded = df_encoded[~((df_encoded < (Q1 - 1.5 * IQR)) | (df_encoded > (Q3 + 1.5 * IQR))).any(axis=1)]
#
#     return df_encoded
# _______________Upload Component_____________________________

@app.callback(
    Output('upload-feedback', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(contents, filename):
    global uploaded_data
    if contents:
        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            uploaded_data = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            return f"Uploaded file: {filename} (Rows: {uploaded_data.shape[0]}, Columns: {uploaded_data.shape[1]})"
        except Exception as e:
            return f"Error reading file: {e}"
    return "No file uploaded."


# _______________Target Selection Component_____________________________

@app.callback(
    Output('target-dropdown', 'options'),
    Input('upload-feedback', 'children')
)
def update_target_dropdown(upload_feedback):
    if uploaded_data is not None:
        numerical_cols = uploaded_data.select_dtypes(include=np.number).columns
        return [{'label': col,  'value': col} for col in numerical_cols]
    return []


# _______________Bar Charts_____________________________
@app.callback(
    [Output('categorical-radio', 'options'),
     Output('category-barchart', 'figure'),
     Output('correlation-barchart', 'figure')],
    [Input('target-dropdown', 'value'),
     Input('categorical-radio', 'value')]
)
def update_barcharts(target, categorical_var):
    if uploaded_data is None or target is None:
        return [], {}, {}

    # Filter numeric columns only for correlation computation
    numeric_data = uploaded_data.select_dtypes(include=[np.number])

    # Categorical columns for dropdown
    cat_cols = uploaded_data.select_dtypes(include=['object']).columns
    categorical_options = [{'label': col, 'value': col} for col in cat_cols]

    # First bar chart (categorical variable vs target)
    if categorical_var:
        avg_values = uploaded_data.groupby(categorical_var)[target].mean()
        category_chart = px.bar(avg_values, x=avg_values.index, y=avg_values.values, title='Average Target by Category')
    else:
        category_chart = {}

    # Second bar chart (correlation of numeric features with target)
    if numeric_data.empty:
        corr_chart = {}
    else:
        correlations = numeric_data.corr()[target].drop(target).abs().sort_values(ascending=False)
        corr_chart = px.bar(correlations, x=correlations.index, y=correlations.values, title='Correlation Strength')

    return categorical_options, category_chart, corr_chart

#---------get checkboxes
#input is the uploaded data,
#output should be the checkboxes of ALL the columns.
@app.callback(
    Output('features-checkboxes', 'options'),
    Input('upload-feedback', 'children')
    #https://chatgpt.com/share/67508cf8-e690-8002-8af3-df68537a651d
)

def get_options(uploaded_feedback):
    if uploaded_data is not None:
        return[{"label": col, "value": col} for col in uploaded_data.select_dtypes(include=np.number).columns ]
    return []

# _______________Train Component_____________________________

#two things need to happen. First, we need to show the check list of all the features in the
@app.callback(
     [Output('model-feedback', 'children')],
    [Input('train-button', 'n_clicks')],
    [State('target-dropdown', 'value'),
     State('features-checkboxes', 'value')]
)
def train_model(n_clicks, target, selected_features):
    if n_clicks is None:
        return ["Please click 'Train Model' to start training."]

    # Debugging: Print target and selected features to see what is being passed
    print(f"Target selected: {target}")
    print(f"Selected features: {selected_features}")

    # If there is no data, target, or selected features, return an error message
    if uploaded_data is None or not target or not selected_features:
        return ["Error: Missing data, target, or features."]

    # Prepare the data for training
    try:
        # Ensure the target and all selected features are in the dataframe
        features_and_target = selected_features + [target]
        X = uploaded_data[selected_features]
        y = uploaded_data[target]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Identify numeric and categorical columns
        numeric_features = X.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Define the preprocessing steps for each type of data
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ]), numeric_features),
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]), categorical_features)
            ])

        # Create the regression pipeline
        global model  # Make sure to update the global model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', XGBRegressor(random_state=43, verbosity=1))
        ])

        # Simplified parameter grid to reduce computational complexity
        param_grid = {
            'regressor__n_estimators': [200, 300],
            'regressor__max_depth': [3, 5, 7],
            'regressor__learning_rate': [0.05, 0.1],
            'regressor__subsample': [0.8, 1.0],
            'regressor__colsample_bytree': [0.8, 1.0],
            'regressor__reg_alpha': [0, 0.1, 1],
            'regressor__reg_lambda': [1, 10]
        }

        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=3,
            scoring='r2',
            verbose=1,
            n_jobs=-1,
            error_score='raise'  # This will help diagnose any fitting errors
        )

        # Fit the grid search
        grid_search.fit(X_train, y_train)

        # Get the best model
        best_model = grid_search.best_estimator_

        # Make predictions
        y_pred = best_model.predict(X_test)

        # Calculate R2 score
        r2_score_final = r2_score(y_test, y_pred)

        return [f"Model trained successfully! R^2 score: {r2_score_final:.4f}"]

    except Exception as e:
        # Catch and print any errors during the training process
        print(f"Error during model training: {e}")
        return [f"Error during model training: {e}"]






# _______________Predict Component_____________________________

@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value')
)
def make_prediction(n_clicks, input_values):
    if model is None or not input_values:
        return "Error: Model not trained or invalid input."
    try:
        features = list(map(float, input_values.split(',')))
        prediction = model.predict([features])
        return f"Predicted value: {prediction[0]:.2f}"
    except Exception as e:
        return f"Error: {e}"


# _______________Run App_____________________________

if __name__ == '__main__':
    app.run_server(debug=True)