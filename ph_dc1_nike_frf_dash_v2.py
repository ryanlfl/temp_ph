######################
## Import Libraries ##
######################

import base64
import io
import math

import urllib
import json
import time

import numpy as np
import pandas as pd

from datetime import timedelta, datetime

import pyodbc
from pandas.io import sql

import pickle

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash_table import DataTable
from dash.dependencies import Input, Output, State

from dash_extensions import Download
from dash_extensions.snippets import send_data_frame

import plotly.graph_objs as go

# For cleaning zones
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

##############
## Settings ##
##############

# Memoize date today
today = pd.to_datetime(datetime.today())

# Use SQL?
use_SQL = True

# Filename of inputs (change to SQL later)
fn = 'Nike-SQL-Result-Sample.csv'

# SQL query for inputs
sql_fn = 'SO Register (Nike).sql'

# Data lake server name
sv_name = 'wmsBIdm.lfapps.net'
# Data lake database name
db_name = 'PH_DATAMART'

# Zoning lookups
lkp_fn_1 = 'Harmonized OTM Zoning.csv'
lkp_fn_2 = 'PH Consignees and Zones.csv'

# App title
app_title = 'PH DC1 Nike Fleet Requirements Forecasting'

# Directory where the data is located
data_path = 'data/'

# Directory where the sql query is located
sql_path = 'sql/'

# Directory where the models are located
model_path = 'models/'

# DOFM model name
dofm_model = '2020-12-09 1226'
# FRF model name
frf_model = '2020-12-10 1741'
# Account
account_name = 'NIKEPH'


# Set number of periods to look back
lookback = 14

# Number of days to forecast
number_of_day = 7

# List of regions
regions = ['LFNCR01',
           'LFVISMIN',
           'LFREG4A',
           'LFREG03',
           'LFRIZ01',
           'LFREG01',
           'LFREG05',
           'LFREG02']

# Variables to be modeled
variables = ['New Quantity', 
             'ConsigneeKey']

# Truck types to be predicted
truck_types = ['AUV', 
               '4WH',
               '6WH',
               '10WH', 
               '20FT',
               '40FT']

# Check if we have stored Data from db to df
isDataFrameLoaded = False
####################
## Load constants ##
####################

# Column order
col_order = pickle.load(open(model_path+'columns(2020-12-08).pkl', 'rb'))

# Instantiate dictionaries to save to
models = {}
models_ub = {}
models_lb = {}
hyperparams = {}

# Reload DOFM models and hyperparameters
for variable in variables:
    models[variable] = {}
    models_ub[variable] = {}
    models_lb[variable] = {}
    hyperparams[variable] = {}
    for region in regions:
        models[variable][region] = pickle.load(open(model_path+dofm_model+' '+variable+'_'+account_name+'_'+region, 'rb'))
        models_ub[variable][region] = pickle.load(open(model_path+dofm_model+' '+variable+'_'+account_name+'_'+region+'_upper', 'rb'))
        models_lb[variable][region] = pickle.load(open(model_path+dofm_model+' '+variable+'_'+account_name+'_'+region+'_lower', 'rb'))
        hyperparams[variable][region] = pd.read_csv(model_path+dofm_model+' '+variable+'_'+account_name+'_'+region+'_hyperparameters.csv')

# Reload FRF models
for truck_type in truck_types:
    models[truck_type] = pickle.load(open(model_path+frf_model+'_'+truck_type+'.pkl', 'rb'))

######################
## Define functions ##
######################



def retry_cdc1_df_loder(times, exceptions, retryAfter=25):
    """
    Retry Decorator
    Retries the wrapped function/method `times` times if the exceptions listed
    in ``exceptions`` are thrown
    :param times: The number of times to repeat the wrapped function/method
    :type times: Int
    :param Exceptions: Lists of exceptions that trigger a retry attempt
    :type Exceptions: Tuple of Exceptions
    :param retryAfter: Wait time in seconds to trigger a retry attempt
    :type retryAfter: int
    """
    def decorator(func):
        def newfn(*args, **kwargs):
            global isDataFrameLoaded
            attempt = 0
            while attempt < times:
                try:
                    if isDataFrameLoaded:
                      return func(*args, **kwargs)
                    raise(Exception('Database taking too long to send data'))
                except exceptions as e:
                    print(f"Exception thrown when attempting to run {func}, attempt {attempt} of {times}\nError --> {str(e)}")       
                    attempt += 1
                    time.sleep(retryAfter)
            return func(*args, **kwargs)
        return newfn
    return decorator

def load_json(jsonData, ori='split'):
    return pd.read_json(jsonData, orient=ori)


# Functions for back-end calculations

def load_data():
    """Load data from CSV. Will change to SQL later.
    """ 
    if use_SQL:
        # Read SQL query 
        file_sql = open(sql_path+sql_fn, 'r')
        sql_query = file_sql.read()
        file_sql.close()

        print('Loading historical data from WMS...')

        # Connect to WMS
        import os

        if os.name=='posix':
            odbcDriver = '{ODBC Driver 17 for SQL Server}'
            trustedConn='UID=dsauser;PWD=Analy*9s;'
        else:
            odbcDriver = '{SQL Server}'
            trustedConn = 'Trusted_Connection=yes;'

        conn = pyodbc.connect('Driver='+odbcDriver+';'
                              'Server='+sv_name+';'
                              'Database='+db_name+';'
                              ''+trustedConn+'Encrypt=no;')
        try:
            # Query data and save to a dataframe
            if conn:
                df = pd.read_sql_query(sql_query, conn)
                print('Historical data loaded!')
                global isDataFrameLoaded
                isDataFrameLoaded = True
            else:
                raise Exception('Unable to Establish Connection')
        except Exception as e:
            # TO-DO Display flash message 
            print(f"ERROR !! --> {e}")
            df = pd.DataFrame({})
        finally:
            # Disconnect from WMS database
            conn.close()

    else:
        # Load CSV
        print('Loading historical data...')
        df = pd.read_csv(data_path+fn,
                         dtype={'ConsigneeKey': 'object'},
                         parse_dates=['AddDate',
                                      'EditDate',
                                      'OrderDate',
                                      'DeliveryDate'])
        # Strip spaces from storer key
        df['StorerKey'] = df['StorerKey'].str.strip()
        print('Historical data loaded!')
    
    return df

def prep_abt(orig_df):
    """Transform the original data set into ABT table by etting the zones.
    orig_df = The original dataframe
    """
    df_abt = orig_df

    print('Cleaning data...')

    # Cleaning round one
    print('\tCleaning using', lkp_fn_1, '...')

    # Read harmonized zoning
    lookup = pd.read_csv(data_path+lkp_fn_1)

    # Use fuzzy matching (Levenshtein distance) to get clean zones from OTM
    temp = dict()
    for city in df_abt['C_City'].unique():
        try:
            temp[city] = process.extractOne(city, lookup['AREA'])[0]
        except:
            pass

    # Get cleaned city names
    df_abt['Cleaned_City'] = df_abt['C_City'].apply(lambda x: temp.get(x))

    # Get zones
    df_abt = df_abt.join(lookup.set_index('AREA')[['Zone 1',
                                                   'Zone 2',
                                                   'Zone 3',
                                                   'Region',
                                                   'Division']],
                     on='Cleaned_City')

    # Cleaning round two
    print('\tCleaning using', lkp_fn_2, '...')

    # Load zones from OTM
    lkp = pd.read_csv(data_path+lkp_fn_2)

    # Remove duplicates
    lkp = lkp.dropna(subset=['ZONE1',
                             'ZONE2',
                             'ZONE3'])
    lkp = lkp.drop_duplicates(subset=['LOCATION_XID'])

    # Join in zones from OTM
    df_abt = df_abt.join(lkp.set_index('LOCATION_XID')[['ZONE1',
                                                        'ZONE2',
                                                        'ZONE3']],
                     on='ConsigneeKey')

    # Where possible, use zones from OTM, else use text analytics
    df_abt['ZONE1'] = df_abt['ZONE1'].fillna(df_abt['Zone 1'])
    df_abt['ZONE2'] = df_abt['ZONE2'].fillna(df_abt['Zone 2'])
    df_abt['ZONE3'] = df_abt['ZONE3'].fillna(df_abt['Zone 3'])

    # Create a hybrid zone that uses divisions for Visayas and Mindanao but ZONE1 for Luzon
    df_abt['Hybrid Zone'] = df_abt['ZONE1']
    for div in ['Visayas', 'Mindanao']:
        #df_abt.loc[df_abt['Division']==div, 'Hybrid Zone'] = div
        df_abt.loc[df_abt['Division']==div, 'Hybrid Zone'] = 'LFVISMIN'

    print('Data cleaned!')

    print('Reshaping data to time series...')

    # Roll up to time series
    ts_data = df_abt.set_index('DeliveryDate').groupby(['StorerKey', 
                                                        'Hybrid Zone']).resample('D')[['New Quantity']].sum()#,
                                                                                       #'New Cube', 
                                                                                       #'New Weight',
                                                                                       #'New Cases']].sum()
    # Include distinct count of consignees
    ts_data = ts_data.join(df_abt.set_index('DeliveryDate').groupby(['StorerKey',
                                                                     'Hybrid Zone']).resample('D')[['ConsigneeKey']].nunique())
    ts_data = ts_data.reset_index()
    ts_data = ts_data.pivot_table(index=['DeliveryDate'],
                                  columns=['StorerKey',
                                           'Hybrid Zone'])
    ts_data = ts_data.fillna(0)

    # Flatten hierarchical index
    ts_data.columns = ['_'.join(col).strip() for col in ts_data.columns.values]

    print('Data reshaped to time series!')

    return ts_data

def dofm(abt):
    """Extend the time series provided by prep_abt() to get the order forecast for the next seven days 
    abt: The analytics base table provided by prep_abt()
    """

    # Initialize final results dataframe
    final = abt.copy()

    # Initialize forecasts dataframe to append to final results
    forecasts = pd.DataFrame(index=pd.date_range(start=abt.index[-1]+timedelta(days=1),
                                                 freq='1D',
                                                 periods=number_of_day))
    print('Calculating DOFM forecasts...')
    # Loop variables to model, with one model per region and variable
    for model_var in variables:
        for region in regions:
            print('\tForecasting', model_var, 'for', region, '...')
            # Prepare dataset by making a copy of the testing predictors
            X_copy = abt.copy()
            # Add columns derived from the date
            X_copy['Date'] = X_copy.index
            X_copy['Weekday'] = X_copy['Date'].dt.dayofweek
            X_copy['Month'] = X_copy['Date'].dt.month
            X_copy = X_copy.join(pd.get_dummies(X_copy['Weekday'], prefix='Weekday'))
            X_copy = X_copy.join(pd.get_dummies(X_copy['Month'], prefix='Month'))
            X_copy = X_copy.drop(['Date', 'Weekday', 'Month'], axis=1)
            # Loop variables as predictors for the selected variable to model
            for variable in variables:
                for i in range(lookback):
                    col = variable+'_'+account_name+'_'+region
                    try:
                        # Create column for previous ith week
                        X_copy[col+'_-'+str(i+1)] = X_copy[col].shift(i+1).fillna(0)
                    except:
                        # Create column of zeroes if no data
                        X_copy[col+'_-'+str(i+1)] = 0
                    # Power transform
                    hp = hyperparams[variable][region]
                    try:
                        power = hp.loc[hp['Variable']==col +'_-'+str(i+1)]['Hyperparameter'].values[0]
                    except:
                        power = 1
                        pass
                    X_copy[col+'_-'+str(i+1)] = X_copy[col +'_-'+str(i+1)].pow(power)
            # Add missing month columns
            for missing in col_order['SKTree'][variable+'_'+account_name+'_'+region]:
                if missing not in X_copy.columns:
                    X_copy[missing] = 0
            X_copy = X_copy[col_order['SKTree'][variable+'_'+account_name+'_'+region]]
            
            # Use model to make predictions using the dataset
            forecasts[model_var+'_'+account_name+'_'+region] = models[model_var][region].predict(X_copy.tail(number_of_day))
            # Remove negative predictions
            forecasts[model_var+'_'+account_name+'_'+region] = forecasts[model_var+'_'+account_name+'_'+region].apply(lambda x: max(x, 0))
            
    # Append forecasts
    final = final.append(forecasts)

    print('DOFM forecasts calculated!')

    return final

def frf(dof, region):
    """Return fleet forecasts based on delivery order forecasts
    dof: The delivery order forecasts
    region: The region to run the forecast in
    """
    # Get data for days to forecast 
    dof = dof.tail(number_of_day)

    print('Calculating FRF forecasts...')

    # Loop through truck types to get predictions for each
    results = pd.DataFrame(index=dof.index)
    for truck_type in truck_types:
        print('\tCalculating', truck_type, 'for', region, '...')
        # Reorder delivery order forecast columns to match expected model inputs (['Ship To', 'Actual QTY Units'])
        X = dof[['ConsigneeKey_'+account_name+'_'+region, 
                 'New Quantity_'+account_name+'_'+region]]
        # Run FRF model
        temp_results = models[truck_type].predict(X)
        # Round off
        temp_results = np.round(temp_results, 0)
        # Save to Pandas DataFrame
        results[truck_type] = pd.Series(temp_results, 
                                        index=dof.index)
        # Replace negative results with 0
        results[truck_type] = results[truck_type].apply(lambda x: max(x, 0))

    print('FRF forecasts calculated!')
    
    return results

# Functions for front-end rendering

def render_table(df, id):
    """Render a dataframe as an HTML table in Dash
    df: The source dataframe
    id: The element ID
    """
    return DataTable(id=id,
                     columns=[{'name': i, 'id': i} for i in df.columns],
                     export_format='xlsx',
                     export_headers='display',
                     data=df.to_dict('records'))

def render_t_table(df, id, colname_column, dateindex_format=False):
    """Render a dataframe as a transposed HTML table in Dash
    df: The source dataframe
    id: The element ID,
    colname_column: The column with entries to be used as column names
    dateindex_format: The format to be used for data columns
    """
    if dateindex_format:
        df[colname_column] = df[colname_column].dt.strftime(dateindex_format)
    df[colname_column] = df[colname_column].apply(str)
    df = df.set_index(colname_column)
    df = df.T
    df = df.reset_index()
    return DataTable(id=id,
                     columns=[{'name': i, 'id': i} for i in df.columns],
                     export_format='xlsx',
                     export_headers='display',
                     data=df.to_dict('records'))

def render_editable_t_table(df, id, colname_column, dateindex_format=False):
    """Render a dataframe as a transposed HTML table in Dash
    df: The source dataframe
    id: The element ID,
    colname_column: The column with entries to be used as column names
    dateindex_format: The format to be used for data columns
    """
    if dateindex_format:
        df[colname_column] = df[colname_column].dt.strftime(dateindex_format)
    df[colname_column] = df[colname_column].apply(str)
    df = df.set_index(colname_column)
    df = df.T
    df = df.reset_index()
    return DataTable(id=id,
                     editable=True,
                     style_data_conditional=[{
                        'if': {'column_editable': True},
                        'backgroundColor': 'rgb(100, 100, 100)',
                        'color': 'white'
                     }],
                     # Need to add code to keep users from overwriting row labels 
                     columns=[{'name': i, 'id': i, 'editable': (i in df.columns)} for i in df.columns],  
                     export_format='xlsx',
                     export_headers='display',
                     data=df.to_dict('records'))

def render_stackedbar(df, y_list=None):
    """Render a dataframe as a time series stacked bar chart
    df: The source dataframe; must have a timeindex
    y_list: The list of time series columns to be plotted
    """

    x_axis = df.index

    if y_list is None:
        y_list = df.columns

    fig_data = []
    for y in y_list:
        fig_data.append(go.Bar(name=y,
                               x=x_axis,
                               y=df[y],
                               text=df[y],
                               textposition='auto'))

    fig = go.Figure(data=fig_data,
                    layout=go.Layout(height=700))

    fig.update_layout(barmode='stack')

    return fig


################
## App Layout ##
################

external_stylesheets = ['assets/style.css']

app = dash.Dash(__name__, url_base_pathname='/cdc1-ph-lrf/',meta_tags=[{"name": "viewport","content": "width=device-width, initial-scale=1"}])
app.title = app_title

from secure_app import SecureApp
sa = SecureApp(app)

app.layout = html.Div(children=[
    dcc.Interval(id='interval-component',
                 interval=24*60*60*1000,  # 1 day in milliseconds
                 n_intervals=0),
    html.Div(id='raw-data', style={'display': 'none'}),
    html.Div(id='abt-data', style={'display': 'none'}),
    html.Div(id='dofm-data', style={'display': 'none'}),
    html.Div(id='frf-data', style={'display': 'none'}),
    html.H1(children=app_title),
    html.Div(
        children='This tool allows TMS planners to estimate future fleet requirements.'),
    dcc.Tabs(value='LFNCR01-tab',
             children=[
                 dcc.Tab(label='LFNCR01',
                         value='LFNCR01-tab',
                         children=[
                             html.Div(id='LFNCR01-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFNCR01-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFNCR01 Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFNCR01-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFNCR01-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFNCR01-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFNCR01-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFNCR01 Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFVISMIN',
                         value='LFVISMIN-tab',
                         children=[
                             html.Div(id='LFVISMIN-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFVISMIN-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFVISMIN Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFVISMIN-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFVISMIN-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFVISMIN-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFVISMIN-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFVISMIN Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFREG4A',
                         value='LFREG4A-tab',
                         children=[
                             html.Div(id='LFREG4A-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFREG4A-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFREG4A Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFREG4A-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFREG4A-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFREG4A-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFREG4A-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFREG4A Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFREG03',
                         value='LFREG03-tab',
                         children=[
                             html.Div(id='LFREG03-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFREG03-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFREG03 Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFREG03-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFREG03-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFREG03-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFREG03-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFREG03 Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFRIZ01',
                         value='LFRIZ01-tab',
                         children=[
                             html.Div(id='LFRIZ01-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFRIZ01-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFRIZ01 Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFRIZ01-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFRIZ01-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFRIZ01-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFRIZ01-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFRIZ01 Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFREG01',
                         value='LFREG01-tab',
                         children=[
                             html.Div(id='LFREG01-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFREG01-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFREG01 Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFREG01-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFREG01-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFREG01-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFREG01-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFREG01 Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFREG05',
                         value='LFREG05-tab',
                         children=[
                             html.Div(id='LFREG05-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFREG05-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFREG05 Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFREG05-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFREG05-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFREG05-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFREG05-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFREG05 Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ]),
                 dcc.Tab(label='LFREG02',
                         value='LFREG02-tab',
                         children=[
                             html.Div(id='LFREG02-vol-div',
                                      className='twelve columns',
                                      #className='six columns',
                                      children=[
                                         html.H2('Volume Forecast'),
                                          html.Div(id='LFREG02-vol-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       render_t_table(pd.DataFrame({'Date': today, 
                                                                                    'LFREG02 Volume Forecast': 0},
                                                                                    index=[0]),
                                                                      'LFREG02-vol-table',
                                                                      'Date',
                                                                      '%b %d %Y')
                                                   ])
                                      ]),
                             html.Div(id='LFREG02-frf-div',
                                      className='twelve columns',
                                      #className='five columns',
                                      children=[
                                         html.H2('Fleet Forecast'),
                                          html.Div(id='LFREG02-frf-container',
                                                   children=[
                                                       # Replace below with data from SQL query
                                                       dcc.Graph(id='LFREG02-chart',
                                                                 figure=render_stackedbar(pd.DataFrame({'LFREG02 Fleet Forecast': 0},
                                                                                                       index=[today])))
                                                   ])
                                      ])
                         ])
             ])
])

sa.requires_authorization_dash(roles=['SCAGROUP'])
###################
## App Callbacks ##
###################
@app.callback(
    [Output('raw-data', 'children'),
     Output('abt-data', 'children')],
    [Input('interval-component', 'n_intervals')]   
    )
def reload_data(n):  
    """Reload data on a regular interval
    n: Count of time intervals
    """ 
    # Load raw data
    df = load_data()


    # Build ABT data from raw data
    abt = prep_abt(df)

    print('ABT\n', abt)

    # Save raw data to div for use later
    df = df.to_json(date_format='iso', 
                    orient='split')

    # Save IMT data to div for use later
    abt = abt.to_json(date_format='iso', 
                      orient='split')

    return df, abt

@app.callback(
    [Output('dofm-data', 'children'),
     Output('LFNCR01-vol-container', 'children'),
     Output('LFVISMIN-vol-container', 'children'),
     Output('LFREG4A-vol-container', 'children'),
     Output('LFREG03-vol-container', 'children'),
     Output('LFRIZ01-vol-container', 'children'),
     Output('LFREG01-vol-container', 'children'),
     Output('LFREG05-vol-container', 'children'),
     Output('LFREG02-vol-container', 'children')],
    [Input('abt-data', 'children')]   
    )
def calc_dofm(abt):  
    """Calculate delivery order forecasts whenever new data comes in
    abt: ABT time series data (from reload_data())
    """ 
    # Load abt data from div
    # abt = pd.read_json(abt, orient='split')
    abt = load_json(abt,'split')


    predictions = dofm(abt)

    # Echo predictions in console
    print('DOFM Predictions\n', predictions)

    # Render table of DOFM predictions per region

    # LFNCR01
    col = 'New Quantity_NIKEPH_LFNCR01'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFNCR01_v = render_editable_t_table(temp,
                                        'LFNCR01-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFVISMIN
    col = 'New Quantity_NIKEPH_LFVISMIN'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFVISMIN_v = render_editable_t_table(temp,
                                        'LFVISMIN-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFREG4A
    col = 'New Quantity_NIKEPH_LFREG4A'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFREG4A_v = render_editable_t_table(temp,
                                        'LFREG4A-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFREG03
    col = 'New Quantity_NIKEPH_LFREG03'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFREG03_v = render_editable_t_table(temp,
                                        'LFREG03-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFRIZ01
    col = 'New Quantity_NIKEPH_LFRIZ01'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFRIZ01_v = render_editable_t_table(temp,
                                        'LFRIZ01-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFREG01
    col = 'New Quantity_NIKEPH_LFREG01'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFREG01_v = render_editable_t_table(temp,
                                        'LFREG01-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFREG05
    col = 'New Quantity_NIKEPH_LFREG05'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFREG05_v = render_editable_t_table(temp,
                                        'LFREG05-vol-table',
                                        'index',
                                        '%b %d %Y')

    # LFREG02
    col = 'New Quantity_NIKEPH_LFREG02'
    temp = predictions[[col]]
    temp = temp.rename(columns={col: 'Nike PH Qty'})
    temp = temp.reset_index().tail(number_of_day).round(0)
    LFREG02_v = render_editable_t_table(temp,
                                        'LFREG02-vol-table',
                                        'index',
                                        '%b %d %Y')

    # Save predictions to div for use later
    predictions = predictions.to_json(date_format='iso', 
                                      orient='split')

    return predictions, LFNCR01_v, LFVISMIN_v, LFREG4A_v, LFREG03_v, LFRIZ01_v, LFREG01_v, LFREG05_v, LFREG02_v

@app.callback(
    Output('LFNCR01-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFNCR01-vol-table', 'data'),
     Input('LFNCR01-vol-table', 'columns')]   
    )
def calc_frf_LFNCR01(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFNCR01'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFVISMIN-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFVISMIN-vol-table', 'data'),
     Input('LFVISMIN-vol-table', 'columns')]   
    )
def calc_frf_LFVISMIN(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFVISMIN'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')

    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFREG4A-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFREG4A-vol-table', 'data'),
     Input('LFREG4A-vol-table', 'columns')]   
    )
def calc_frf_LFREG4A(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFREG4A'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFREG03-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFREG03-vol-table', 'data'),
     Input('LFREG03-vol-table', 'columns')]   
    )
def calc_frf_LFREG03(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFREG03'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFRIZ01-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFRIZ01-vol-table', 'data'),
     Input('LFRIZ01-vol-table', 'columns')]   
    )
def calc_frf_LFRIZ01(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFRIZ01'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFREG01-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFREG01-vol-table', 'data'),
     Input('LFREG01-vol-table', 'columns')]   
    )
def calc_frf_LFREG01(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFREG01'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFREG05-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFREG05-vol-table', 'data'),
     Input('LFREG05-vol-table', 'columns')]   
    )
def calc_frf_LFREG05(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFREG05'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]

    return output

@app.callback(
    Output('LFREG02-frf-container', 'children'),
    [Input('dofm-data', 'children'),
     Input('LFREG02-vol-table', 'data'),
     Input('LFREG02-vol-table', 'columns')]   
    )
def calc_frf_LFREG02(dof, rows, cols):  
    """Calculate fleet requirements forecasts whenever new delivery order forecasts come in 
    dof: Delivery order forecast data (from dofm())
    rows: Data from user input table
    cols: Column labels from user input table
    """ 
    region = 'LFREG02'
    # Load abt data from div
    # dof = pd.read_json(dof, orient='split')
    dof = load_json(dof,'split')


    #print('Div data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Load user inputs
    if cols != [{'name': 'index', 'id': 'index'}, {'name': today.strftime('%b %d %Y'), 'id': today.strftime('%b %d %Y')}]:
        # Load data from user-editable table
        table_input = pd.DataFrame(rows, columns=[c['name'] for c in cols])

        # Transpose data
        table_input = table_input.T
        # Reset index
        table_input = table_input.reset_index()
        # Get columns from first row
        table_input.columns = list(table_input.loc[0])
        # Drop first row
        table_input = table_input[1:]
        # Convert to date format
        table_input['index'] = pd.to_datetime(table_input['index'])
        # Set date to index
        table_input = table_input.set_index('index')

        #print('Table data:')
        #print(table_input)

        # Overwrite data with user inputs
        for index, row in table_input.iterrows():
            dof.loc[index, 'New Quantity_NIKEPH_'+region] = row['Nike PH Qty']

    #print('Final data:')
    #print(dof['New Quantity_NIKEPH_'+region])

    # Calculate FRF
    results = frf(dof, region)

    # Render chart of results
    chart = render_stackedbar(results, y_list=truck_types)
    chart = dcc.Graph(id=region+'-chart',
                      figure=chart)

    # Render table of results
    table = render_t_table(results.reset_index(),
                           region+'-frf-table',
                           'index',
                           '%b %d %Y')

    output = [chart, table]
    
    return output

################
## Run Server ##
################

if __name__ == '__main__':
    #app.run_server(debug=True, port=8084, host='0.0.0.0')
    app.run_server(debug=True, port=8004, host='127.0.0.1')
