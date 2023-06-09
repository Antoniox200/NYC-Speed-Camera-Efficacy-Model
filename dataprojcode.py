"""
    Name: Antonio Iadicicco
    Email: antonio.iadicicco90@myhunter.cuny.edu
    Resources:  w3schools, Python documentation, Geekforgeeks, Stackoverflow, Pandas.pydata.org,
    prophet documentation, rapidfuzz documentation, geopy documentation, folium documentation,
    tqdm documentation, matplotlib documentation, geopy documentation
"""

import pandas as pd
import numpy as np
from geopy.geocoders import GoogleV3
from geopy.exc import GeocoderTimedOut
import folium
from rapidfuzz import fuzz, process
import re
from tqdm import tqdm
import concurrent.futures
from functools import partial
import re
from geopy.distance import great_circle
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric


api_key = 'GOOGLE_API_KEY_GOES_HERE'
# Initialize Google Geocoder
geolocator = GoogleV3(api_key=api_key)

'''
Speed Camera Data Science Term Project

This file contains the code for the data processing, analysis, and predictive model for the speed camera and crash data.
All of the code was originally written in Jupyter Notebooks,
but was moved and reorganized into a .py file

'''

###########################################################################################
'''
Initial processing of the speed camera ticket data. This section handles concatenating the dataframes, removing duplicates,
and some initial cleaning of the data.

This data is used to build the key dataframe of speed camera locations.
'''

def process_csv2014(file_path, chunk_size=10000):
    # Create an empty DataFrame to store the processed data
    result = pd.DataFrame()
    print("Processing file: " + file_path)

    # Loop over the chunks of data
    for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunk_size):

        # Filter rows where "violation_description" is not "PHTO SCHOOL ZN SPEED VIOLATION"
        chunk = chunk[chunk['Violation Description'] == 'PHTO SCHOOL ZN SPEED VIOLATION']

        # Create new "street" column by concatenating "street_name" and "intersecting_street"
        chunk['street'] = chunk['Street'] + chunk['Intersecting Street']
        chunk['key'] = chunk['Street'] + chunk['Intersecting Street']


        # Remove rows with duplicate "street" values
        chunk.drop_duplicates(subset='street', keep='first', inplace=True)

        # Append the processed chunk to the result DataFrame
        result = pd.concat([result, chunk], ignore_index=True)

    # Remove rows with duplicate "street" values
    result.drop_duplicates(subset='street', keep='first', inplace=True)

    #rename street column to "Street Name"
    result.rename(columns={'Street': 'Street Name'}, inplace=True)

    # Return the modified DataFrame
    return result

def process_csv(file_path, chunk_size=10000):
    # Create an empty DataFrame to store the processed data
    result = pd.DataFrame()
    print("Processing file: " + file_path)

    # Loop over the chunks of data
    for chunk in pd.read_csv(file_path, low_memory=False, chunksize=chunk_size):

        # Filter rows where "violation_description" is not "PHTO SCHOOL ZN SPEED VIOLATION"
        chunk = chunk[chunk['Violation Description'] == 'PHTO SCHOOL ZN SPEED VIOLATION']

        # Create new "street" column by concatenating "street_name" and "intersecting_street"
        chunk['street'] = chunk['Street Name'] + chunk['Intersecting Street']
        chunk['key'] = chunk['Street Name'] + chunk['Intersecting Street']


        # Remove rows with duplicate "street" values
        chunk.drop_duplicates(subset='street', keep='first', inplace=True)

        # Append the processed chunk to the result DataFrame
        result = pd.concat([result, chunk], ignore_index=True)

    # Remove rows with duplicate "street" values
    result.drop_duplicates(subset='street', keep='first', inplace=True)
    # Return the modified DataFrame
    return result


def concat_df(df1, df2):
    # Concatenate df1 and df2
    df = pd.concat([df1, df2], ignore_index=True)

    #remove rows with duplicate "street" values
    df.drop_duplicates(subset='street', keep='first', inplace=True)

    # Return the concatenated DataFrame
    return df

def add_at_sign(df):

    # Loop through rows and check if both "street_name" and "intersecting_street" don't contain "@"
    for index, row in df.iterrows():
        if '@' not in row['Street Name'] and '@' not in row['Intersecting Street']:
            # Add "@" at the end of the "street_name" column for that row
            df.at[index, 'Street Name'] = row['Street Name'] + '@'
            #concatenate "street_name" and "intersecting_street" and store in "street" column
            df.at[index, 'street'] = row['Street Name'] + row['Intersecting Street']

    # Return the modified DataFrame
    return df

def convert_at_to_and(df):

    df['street'] = df['street'].str.replace('@', '&')
    return df

#process and conctenate the files with the names corresponding to the years 2014-2023
def process_concatenate():
    file_names = [
        "2014.csv",
        "2015.csv",
        "2016.csv",
        "2017.csv",
        "2018.csv",
        "2019.csv",
        "2020.csv",
        "2021.csv",
        "2022.csv",
        "2023.csv",
    ]
    dataframes = []

    for file_name in file_names:
        if file_name == "2014.csv":
            df = process_csv2014(file_name)
        else:
            df = process_csv(file_name)
        print(f"{file_name} processed")
        dataframes.append(df)

    concatenated_df = pd.concat(dataframes, ignore_index=True)
    print("Concatenation complete")
    return concatenated_df


def executeSection1():
    print("Starting program")
    # Call process_csv() function
    df = process_concatenate()

    #open key.csv
    df = pd.read_csv("key.csv")
    # Call add_at_sign() function
    df = add_at_sign(df)

    df = convert_at_to_and(df)
    # Save the modified DataFrame to CSV file
    df.to_csv("key.csv", index=False)

###########################################################################################
'''
This section is for geocoding the speed camera locations. It uses the Google Geocoding API to convert the addresses to
latitude and longitude coordinates. This  is done in order to eventually match the camera locations to their corresponding
collision locations.
'''

# Function to geocode an address using Google Geocoder
def geocode_address(address, boro):
    try:
        print("Geolocating Address:", address + boro)
        location = geolocator.geocode(components={'route': address + boro, 'locality': 'New York', 'administrative_area': 'NY', 'country': 'US'})
        print("Geocoded Address:", location)
        print("Latitude:", location.latitude)
        print("Longitude:", location.longitude)
        return location.latitude, location.longitude
    except Exception as e:
        print(f"Error geocoding address: {address} - {e}")
        return None, None


def add_boro(boro):
    if boro == "QN":
        return ",Queens"
    elif boro == "BK":
        return ",Brooklyn"
    elif boro == "BX":
        return ",Bronx"
    elif boro == "ST":
        return ",Staten Island"
    elif boro == "MN":
        return ",Manhattan"
    else:
        return ""


def create_map(df):
    #create map
    m = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
    for index, row in df.iterrows():
        if row['coordinates'] != None:
            folium.Marker([row['latitude'], row['longitude']], popup=row['street']).add_to(m)

    m.save('Map3.html')
    return m

def executeSection2():
    # Read the CSV file into a DataFrame
    csv_file_path = 'key.csv'  # Replace with the path to your CSV file
    df = pd.read_csv(csv_file_path)

    # Create new columns for latitude and longitude
    df['latitude'] = None
    df['longitude'] = None
    df['coordinates'] = None

    #convert street column to string
    df['street'] = df['street'].astype(str)
    #convert boro column to string
    df['Violation County'] = df['Violation County'].astype(str)

    # Geocode the 'key' column and store the results in the new columns
    for index, row in df.iterrows():
        address = row['street']
        boro = row['Violation County']
        boro = add_boro(boro)
        latitude, longitude = geocode_address(address, boro)
        df.at[index, 'latitude'] = latitude
        df.at[index, 'longitude'] = longitude
        df.at[index, 'coordinates'] = (latitude, longitude)

    # Save the DataFrame with geocoded data to a new CSV file
    output_csv_file_path = 'geocoded_data.csv'
    df.to_csv(output_csv_file_path, index=False)
    print("Geocoding complete. Results saved to", output_csv_file_path)

    #create map
    m = create_map(df)
    #save map
    m.save('CameraMap.html')

###########################################################################################
'''
This section is for creating the time snapshots of the given camera locations, with their
corresponding ticket counts.
'''

def read_key_file(filename):
    """
    Reads in the key file and creates a 'year' column based on the 'first_observed' column.
    """
    key = pd.read_csv(filename)
    key['year'] = key['first_observed'].str[-4:]
    return key

def create_new_df(key):
    """
    Creates a new dataframe with a row for each month for each item in the key.
    """
    items = key['key']
    date_rng = pd.date_range(start='01/2014', end='02/2023', freq='MS')
    new_df = pd.DataFrame(columns=['item', 'date'])
    for item in items:
        for date in date_rng:
            new_df = new_df.append({'item': item, 'date': date}, ignore_index=True)
    return new_df


def update_new_file(new_df, key):
    """
    Updates the new file by adding 'street' and 'first_observed' columns based on a match between the 'item'
    column in the new file and the 'key' column in the key file.
    """
    for index, row in new_df.iterrows():
        for key_index, key_row in key.iterrows():
            if row['item'] == key_row['key']:
                new_df.loc[index, 'street'] = key_row['street']
                new_df.loc[index, 'first_observed'] = key_row['first_observed']
    return new_df

def executeSection3():
    # Read in the key file and create a 'year' column
    key = read_key_file('key.csv')

    # Create a new dataframe with a row for each month for each item in the key
    new_df = create_new_df(key)

    # Update the new file by adding 'street' and 'first_observed' columns
    new_df = update_new_file(new_df, key)

    # Save the updated file
    new_df.to_csv('KeyTest2.csv', index=False)

##############################################################################################################
'''
This section creates ticket counts for each month for each street in the key file.
'''

def load_and_rename_columns(file_name):
    '''
    Loads the file and renames the columns to match the other files.
    '''
    df = pd.read_csv(file_name)
    df.rename(columns={'Street': 'Street Name', 'Issue Date': 'Date'}, inplace=True)
    return df

def concatenate_dataframes(dfs):
    '''
    Concatenates the dataframes into one dataframe.
    '''
    return pd.concat(dfs, ignore_index=True)

def preprocess_dataframe(df):
    '''
    Preprocesses the dataframe by removing unnecessary columns, filtering for only speed camera violations,
    and creating a 'street' column.
    '''
    df['street'] = df['Street Name'] + df['Intersecting Street']
    df = df[df['Violation Description'] == 'PHTO SCHOOL ZN SPEED VIOLATION']
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    return df

def group_and_aggregate(df):
    '''
    Groups the dataframe by street, month, and year and aggregates the count of violations.
    '''
    df = df.groupby(['street', 'month', 'year'])['count'].sum().reset_index()
    df['Date'] = df['month'].astype(str) + '/' + df['year'].astype(str)
    return df

def fill_missing_months(df):
    '''
    Fills in missing months with a count of 0.
    '''
    df2 = df.set_index(['street_x', 'Date']).unstack(fill_value=0).stack().reset_index()
    df2 = pd.merge(df2, df[['street_x', 'latitude', 'longitude', 'coordinates', 'first_observed', 'street_y']], how='left', left_on='street_x', right_on='street_x')
    return df2

def executeSection4():
    file_names = ['2014tickets.csv', '2015tickets.csv', '2016tickets.csv', '2017tickets.csv', '2018tickets.csv', '2019tickets.csv', '2020tickets.csv', '2021tickets.csv', '2022tickets.csv', '2023tickets.csv']
    dataframes = [load_and_rename_columns(file_name) for file_name in file_names]
    print("Files opened")

    df = concatenate_dataframes(dataframes)
    print("Files concatenated")

    df = preprocess_dataframe(df)
    df = group_and_aggregate(df)
    df = fill_missing_months(df)

    df.to_csv('ticketform.csv', index=False)
    print("File saved")




##############################################################################################################
'''
This section is for adding monthly ticket counts to the time snapshots of the speed camera
intersections created in the previous section.
'''


def read_files():
    '''
    reads in the key file and the ticket count file
    '''
    key = pd.read_csv('geocoded_data.csv')
    df = pd.read_csv('ticketform.csv')
    return key, df

def create_dataframe(key):
    '''
    creates a dataframe with a row for each month for each street in the key file
    '''
    df2 = []
    for _, street_dict in key.iterrows():
        street = street_dict['street']
        for year in range(2012, 2024):
            for month in range(1, 13):
                df2.append([
                    street, month, year,
                    street_dict['latitude'], street_dict['longitude'],
                    street_dict['coordinates'], street_dict['first_observed'],
                    street_dict['key']
                ])
    df2 = pd.DataFrame(df2, columns=['street', 'month', 'year', 'latitude', 'longitude', 'coordinates', 'first_observed', 'key'])
    return df2

def merge_dataframes(df2, df):
    '''
    merges the ticket count file with the dataframe created in the previous function
    '''
    df2 = pd.merge(df2, df, how='left', left_on=['key', 'month', 'year'], right_on=['street', 'month', 'year'])
    return df2

def process_dataframe(df2):
    '''
    processes the dataframe by filling in missing months and creating a 'hascamera' column
    '''
    df2['hascamera'] = np.where(df2['count'].isnull(), False, True)
    df2['Date'] = df2['month'].astype(str) + '/' + df2['year'].astype(str)

    df2['street_x'] = df2['street_x'].str.replace('+', '&')
    df2 = df2[df2['street_x'].str.contains('&') == True]

    if 'street_x' in df2.columns and not df2['street_x'].empty:
        if any(df2['street_x'].str.contains('&')):
            temp_df = df2['street_x'].str.split('&', expand=True)
            df2['street1'] = temp_df[0]
            df2['street2'] = temp_df[1]
            df2['street1'] = df2['street1'].str[3:]
        else:
            print("No '&' character found in 'street_x' column for splitting")
    else:
        print("Column 'street_x' does not exist or is empty")

    return df2

def executeSection5():
    #read key and ticket count files
    key, df = read_files()
    #create dataframe with a row for each month for each street in the key file
    df2 = create_dataframe(key)
    #merge ticket count file with the dataframe created in the previous function
    df2 = merge_dataframes(df2, df)
    #process the dataframe by filling in missing months and creating a 'hascamera' column
    df2 = process_dataframe(df2)
    #save the dataframe
    df2.to_csv('dateform2RevisionXX.csv', index=False)

##############################################################################################################
'''
This section is for narrowing down/cleaning the large collision dataset to only include collisions that occurred
at the same intersection as the camera locations.
'''


def preprocess_street_names(df, columns):
    # Convert to lowercase
    df[columns] = df[columns].apply(lambda x: x.str.lower())

    # Remove extra whitespace and punctuation
    df[columns] = df[columns].apply(lambda x: x.str.replace(r'\s+', ' ', regex=True).str.strip())

    # Replace common abbreviations and directional indicators
    abbreviations = {'st.': 'street', 'rd.': 'road', 'blvd.': 'boulevard', ' dr.': 'drive', ' pl.': 'place', ' ave.': 'avenue', 'st.': 'street', ' rd.': 'road', 'blvd': 'boulevard', ' dr.': 'drive'}
    for col in columns:
        for abbr, full in abbreviations.items():
            df[col] = df[col].apply(lambda x: re.sub(f"{abbr}\\b", full, x))

    return df

def match_on_and_cross_street(args):
    row, key_df, threshold = args
    on_street, cross_street = row['ON STREET NAME'], row['CROSS STREET NAME']
    for _, key_row in key_df.iterrows():
        street_name_ratio = fuzz.token_set_ratio(on_street, key_row['Street Name'])
        intersecting_street_ratio = fuzz.token_set_ratio(cross_street, key_row['Intersecting Street'])

        if street_name_ratio >= threshold and intersecting_street_ratio >= threshold:
            return True

def parallel_matching(data_df, key_df, num_workers=8):
    def update_progress_bar(future_result):
        nonlocal pbar
        pbar.update()

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        args = [(row, key_df, 80) for _, row in data_df.iterrows()]

        with tqdm(total=len(args), desc="Matching streets") as pbar:
            futures = []
            for arg in args:
                future = executor.submit(match_on_and_cross_street, arg)
                future.add_done_callback(update_progress_bar)
                futures.append(future)

            results = [bool(future.result()) for future in futures]

    return results


def executeSection6():
    # Load the CSV files
    data_file = 'crashes.csv'
    key_file = 'key.csv'

    data_df = pd.read_csv(data_file)
    key_df = pd.read_csv(key_file)

    # Convert all rows in data_df and key_df to string
    data_df = data_df.astype(str)
    key_df = key_df.astype(str)

    # Preprocess street names in both dataframes
    data_df = preprocess_street_names(data_df, ['ON STREET NAME', 'CROSS STREET NAME'])
    key_df = preprocess_street_names(key_df, ['Street Name', 'Intersecting Street'])

    # Initialize a new column for match status
    data_df['match_status'] = False

    # Combine the street names in key_df
    key_df['combined_street_name'] = key_df['Street Name'] + ' ' + key_df['Intersecting Street']

    # Match ON STREET and CROSS STREET, and update match_status
    data_df['match_status'] = parallel_matching(data_df, key_df)

    # Drop rows without a suitable match
    data_df = data_df[data_df['match_status']]

    # Save the results to a new CSV file
    data_df.to_csv('FUZZEDSTREETS.csv', index=False)

##############################################################################################################
'''
This section is for further narrowing down/cleaning the large collision dataset to only include collisions that occurred
within a certain distance of the camera locations (1/8 Mile).

If a collision occurred within 1/8 mile of a camera location,
It is added to the time snapshot of the camera location for that month, along with injuries and fatalities.

'''

def preprocess_street_names(df, columns):
    abbreviations = {'st.': 'street', 'rd.': 'road', 'blvd.': 'boulevard', ' dr.': 'drive', ' pl.': 'place', ' ave.': 'avenue', 'st.': 'street', ' rd.': 'road', 'blvd': 'boulevard', ' dr.': 'drive'}

    def replace_abbreviations(s):
        if isinstance(s, str):
            for abbr, full in abbreviations.items():
                s = re.sub(f"{abbr}\\b", full, s)
        return s

    for col in columns:
        df[col] = df[col].apply(replace_abbreviations)

    return df

def executeSection7():
    # Load the CSV files
    data_file = 'FUZZEDSTREETS.csv'
    key_file = 'dateform2RevisionXX.csv'

    data_df = pd.read_csv(data_file)
    key_df = pd.read_csv(key_file)

    # Preprocess street names in the data and key dataframes
    data_df = preprocess_street_names(data_df, ['ON STREET NAME', 'CROSS STREET NAME'])
    key_df = preprocess_street_names(key_df, ['Camera Location'])

    # Initialize new columns for Crashes, Injuries, Deaths, and matched in the data dataframe
    key_df['Crashes'] = 0
    key_df['Injuries'] = 0
    key_df['Deaths'] = 0
    data_df['matched'] = False

    # Convert crash date to datetime object and create month-year column
    data_df['CRASH DATE'] = pd.to_datetime(data_df['CRASH DATE'])
    data_df['month-year'] = data_df['CRASH DATE'].dt.to_period('M')

    # Group data dataframe by month-year
    data_grouped = data_df.groupby('month-year')

    # Iterate through rows in the key dataframe with tqdm progress bar
    for key_idx, key_row in tqdm(key_df.iterrows(), total=key_df.shape[0], desc="Processing rows"):
        key_month, key_year = int(key_row['month']), int(key_row['year'])
        key_location = (key_row['latitude'], key_row['longitude'])

        try:
            # Get rows in the data dataframe for the same month and year
            data_rows = data_grouped.get_group(pd.Period(f"{key_year}-{key_month}"))
        except KeyError:
            continue

        # Filter unmatched rows and calculate distance for each row
        data_rows = data_rows[data_rows['matched'] == False]
        data_rows['distance'] = data_rows.apply(lambda row: great_circle((row['LATITUDE'], row['LONGITUDE']), key_location).miles, axis=1)

        # Filter rows within 1/8 mile
        filtered_rows = data_rows[data_rows['distance'] <= 1/8]

        # Mark filtered rows as matched
        data_df.loc[filtered_rows.index, 'matched'] = True

        # Increment Crashes, Injuries, and Deaths in the key dataframe
        key_df.loc[key_idx, 'Crashes'] += filtered_rows.shape[0]
        key_df.loc[key_idx, 'Injuries'] += filtered_rows['NUMBER OF PERSONS INJURED'].sum()
        key_df.loc[key_idx, 'Deaths'] += filtered_rows['NUMBER OF PERSONS KILLED'].sum()

    # Save the results to new CSV files
    data_df.to_csv('MATCHED_DATA.csv', index=False)
    key_df.to_csv('finalData.csv', index=False)

##############################################################################################################
'''
This section uses the final dataset to train a model to predict the number of tickets, crashes, injuries, and deaths
Along with the visualizations of the results.

This section also performs model evaluations to determine the accuracy of the model, along with
some other summary statistics.
'''
def read_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'], format='%b-%y')

    # Convert date to MM/YYYY
    data['Date'] = data['Date'].dt.strftime('%m/%Y')

    # Calculate the total collisions, tickets, injuries, and deaths for each date
    agg_data = data.groupby('Date').agg({'tickets given': 'sum', 'Crashes': 'sum', 'Injuries': 'sum', 'Deaths': 'sum'}).reset_index()

    # Rename columns to match the format required by Prophet
    agg_data.rename(columns={'Date': 'ds', 'tickets given': 'y_tickets', 'Crashes': 'y_crashes', 'Injuries': 'y_injuries', 'Deaths': 'y_deaths'}, inplace=True)

    # Set the 'ds' column as index and ensure it's a DatetimeIndex
    agg_data.set_index(pd.DatetimeIndex(agg_data['ds']), inplace=True)
    # Resample the data to a monthly frequency and fill missing values with zeros
    agg_data_monthly = agg_data.resample('M').sum().reset_index()

    return agg_data_monthly

# Function to create and fit a Prophet model
def create_and_fit_prophet(data, column_name):
    prophet_data = data[['ds', column_name]].rename(columns={column_name: 'y'})
    prophet_model = Prophet(weekly_seasonality=False, daily_seasonality=False)
    prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    prophet_model.fit(prophet_data)
    return prophet_model

def train_prophet_models(agg_data_monthly):
    # Train the Prophet models for tickets, crashes, injuries, and deaths
    tickets_prophet = create_and_fit_prophet(agg_data_monthly, 'y_tickets')
    crashes_prophet = create_and_fit_prophet(agg_data_monthly, 'y_crashes')
    injuries_prophet = create_and_fit_prophet(agg_data_monthly, 'y_injuries')
    deaths_prophet = create_and_fit_prophet(agg_data_monthly, 'y_deaths')

    return tickets_prophet, crashes_prophet, injuries_prophet, deaths_prophet

# Function to make predictions using a trained Prophet model
def make_predictions(model, future_dataframe):
    return model.predict(future_dataframe)

# Function to plot the forecast
def plot_forecast(model, forecast, title):
    fig = model.plot(forecast)
    plt.title(title)
    plt.show()

def plot_all_forecasts(tickets_prophet, crashes_prophet, injuries_prophet, deaths_prophet, future):
    # Predict the future number of tickets, crashes, injuries, and deaths
    tickets_forecast = make_predictions(tickets_prophet, future)
    crashes_forecast = make_predictions(crashes_prophet, future)
    injuries_forecast = make_predictions(injuries_prophet, future)
    deaths_forecast = make_predictions(deaths_prophet, future)

    # Plot the forecast for tickets, crashes, injuries, and deaths
    plot_forecast(tickets_prophet, tickets_forecast, 'Tickets Forecast')
    plot_forecast(crashes_prophet, crashes_forecast, 'Crashes Forecast')
    plot_forecast(injuries_prophet, injuries_forecast, 'Injuries Forecast')
    plot_forecast(deaths_prophet, deaths_forecast, 'Deaths Forecast')

def plot_avg_crashes_per_month(data):
    # Continue with the bar chart for average crashes per month at camera locations
    avg_crashes = data.groupby('hascamera')['Crashes'].mean()

    # Create a bar chart to compare average crashes per month for camera locations
    fig, ax = plt.subplots()
    ax.bar(avg_crashes.index, avg_crashes.values)

    # Set labels and title
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Before Camera', 'After Camera'])
    ax.set_ylabel('Average Crashes per Month')
    ax.set_title('Average Crashes per Month at Camera Locations')

    # Display the bar chart
    plt.show()

# Function to perform cross-validation and plot performance metrics
def evaluate_model(model, horizon, initial=None, period=None):
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_p = performance_metrics(df_cv, rolling_window=1)
    print(df_p)

    # Plot the performance metrics
    for metric in ['mae', 'mse', 'rmse']:
        plot_cross_validation_metric(df_cv, metric=metric)
        plt.title(f'{metric.upper()} Performance')
        plt.show()


def executeSection8():
    # Read and preprocess data
    file_path = 'fullData.csv'
    agg_data_monthly = read_and_preprocess_data(file_path)

    # Train the Prophet models
    tickets_prophet, crashes_prophet, injuries_prophet, deaths_prophet = train_prophet_models(agg_data_monthly)

    # Define the period for which you want to make predictions (12 months by default)
    future = tickets_prophet.make_future_dataframe(periods=12, freq='M')

    # Plot all forecasts
    plot_all_forecasts(tickets_prophet, crashes_prophet, injuries_prophet, deaths_prophet, future)

    # Read the original data again to plot the bar chart
    data = pd.read_csv(file_path)
    plot_avg_crashes_per_month(data)

    # Evaluate the accuracy of the tickets model
    evaluate_model(tickets_prophet, horizon='180 days', initial='730 days', period='180 days')


##############################################################################################################

def main():
    '''
    This is the main function that executes all the sections of the program.

    Warning: The full program takes a very long time to run. It is recommended to run each section individually.
    The full program can take up to 10 hours to run on my local machine.
    Certain sections have tqdm progress bars to show progress.
    '''
    #iniial data processing and cleaning
    executeSection1()
    #geocode camera locations
    executeSection2()
    #create time snapshots of camera locations from 2012-2023
    executeSection3()
    #create ticket counts for each camera location
    executeSection4()
    #fill in ticket counts for each camera location/date
    executeSection5()
    #initial cleaning/fuzzing of collsion dataset (takes a long time to run)
    executeSection6()
    #create collision counts for each camera location (takes a long time to run)
    executeSection7()
    #create predictions and visualizations
    executeSection8()

if __name__ == "__main__":
    main()
