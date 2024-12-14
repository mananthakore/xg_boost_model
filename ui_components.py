from dash import dcc, html
import dash_bootstrap_components as dbc

def create_layout():
    return dbc.Container([
        # Header
        dbc.Row(dbc.Col(html.H1("Regression Prediction App", className="text-center my-4"))),

        # Upload Dataset Section
        dbc.Row([
            dbc.Col([
                html.H2("1. Upload Dataset"),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Drag and Drop or ', html.A('Select a File')]),
                    style={
                        'width': '100%', 'height': '60px', 'lineHeight': '60px',
                        'borderWidth': '1px', 'borderStyle': 'dashed',
                        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='upload-feedback', className='text-info'),
            ], width=6),
        ]),

        # Target Variable Selection
        dbc.Row([
            dbc.Col([
                html.H2("2. Select Target Variable"),
                dcc.Dropdown(id='target-dropdown', placeholder='Select target variable'),
            ], width=6),
        ]),

        # Data Analysis Section
        dbc.Row([
            dbc.Col([
                html.H2("3. Analyze Data"),
                dbc.Card([
                    dbc.CardBody([
                        dcc.RadioItems(id='categorical-radio'),
                        dcc.Graph(id='category-barchart'),
                        dcc.Graph(id='correlation-barchart'),
                    ])
                ])
            ], width=12)
        ]),

        # Model Training Section
        dbc.Row([
            dbc.Col([
                html.H2("4. Train Model"),
                dcc.Checklist(id='features-checkboxes'),
                dbc.Button('Train Model', id='train-button', color="primary", className="mt-3"),
                html.Div(id='model-feedback', className='text-success mt-3'),
            ], width=12)
        ]),

        # Prediction Section
        dbc.Row([
            dbc.Col([
                html.H2("5. Make Predictions"),
                dbc.Input(id='predict-input', placeholder="Enter feature values...", type="text"),
                dbc.Button('Predict', id='predict-button', color="secondary", className="mt-3"),
                html.Div(id='prediction-output', className='text-warning mt-3'),
            ], width=12)
        ])
    ], fluid=True)
