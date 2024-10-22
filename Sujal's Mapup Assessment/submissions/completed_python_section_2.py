import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform

# Function to calculate distance matrix (assuming distance data is already provided)


def calculate_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame): A dataframe where each row represents a data point.

    Returns:
        pandas.DataFrame: Distance matrix, where the value at (i, j) is the distance between row i and row j.
    """
    # Calculate the distance matrix using pdist and squareform
    distances = pdist(df, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Convert to DataFrame for easier handling and return
    return pd.DataFrame(distance_matrix, index=df.index, columns=df.index)

# Function to unroll distance matrix into long format
def unroll_distance_matrix(df) -> pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    unrolled_data = []
    for idx, row in df.iterrows():
        id_start = row.name
        for col in df.columns:
            id_end = col
            distance = row[col]
            if id_start != id_end:  # Avoid self-distances
                unrolled_data.append([id_start, id_end, distance])

    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])
    return unrolled_df

# Function to find IDs within 10% of the reference ID's average distance
def find_ids_within_ten_percentage_threshold(df, reference_id) -> pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    avg_distances = df.groupby('id_start')['distance'].mean()
    ref_avg_distance = avg_distances[reference_id]

    lower_bound = ref_avg_distance * 0.9
    upper_bound = ref_avg_distance * 1.1

    matching_ids = avg_distances[(avg_distances >= lower_bound) & (avg_distances <= upper_bound)].index.tolist()

    return df[df['id_start'].isin(matching_ids)]

# Function to calculate toll rates for different vehicle types
def calculate_toll_rate(df) -> pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    toll_data = []
    for _, row in df.iterrows():
        id_start, id_end, distance = row['id_start'], row['id_end'], row['distance']
        # Sample toll rate calculations for various vehicle types (can be modified)
        moto = distance * 0.8
        car = distance * 1.2
        rv = distance * 1.5
        bus = distance * 2.2
        truck = distance * 3.6
        toll_data.append([id_start, id_end, moto, car, rv, bus, truck])

    toll_df = pd.DataFrame(toll_data, columns=['id_start', 'id_end', 'moto', 'car', 'rv', 'bus', 'truck'])
    return toll_df

# Function to calculate time-based toll rates for different time intervals
def calculate_time_based_toll_rates(df) -> pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    time_based_tolls = []
    for _, row in df.iterrows():
        id_start, id_end, distance, start_day, start_time, end_day, end_time = row['id_start'], row['id_end'], row['distance'], row['start_day'], row['start_time'], row['end_day'], row['end_time']
        # Sample time-based toll rates (can be adjusted)
        moto = distance * 0.65
        car = distance * 0.96
        rv = distance * 1.2
        bus = distance * 1.75
        truck = distance * 2.88
        time_based_tolls.append([id_start, id_end, distance, start_day, start_time, end_day, end_time, moto, car, rv, bus, truck])

    time_based_toll_df = pd.DataFrame(time_based_tolls, columns=['id_start', 'id_end', 'distance', 'start_day', 'start_time', 'end_day', 'end_time', 'moto', 'car', 'rv', 'bus', 'truck'])
    return time_based_toll_df
