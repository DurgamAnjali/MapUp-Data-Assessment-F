import pandas as pd
import networkx as nx

def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    G = nx.Graph()

    # Add edges and distances to the graph
    for index, row in df.iterrows():
      G.add_edge(row['id_start'], row['id_end'], distance=row['distance'])
    
    # Calculate the shortest path lengths between all pairs of nodes
      distance_matrix = nx.floyd_warshall_numpy(G, weight='distance')

    # converting distance matrix to dataframe
    df = pd.DataFrame(distance_matrix, index=G.nodes, columns=G.nodes)

    # replacing nan values with 0 in diagonal position
    df = df.fillna(0)


    return df


def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here

    unrolled_df = []
    # Iterate over each row in the distance_df
    for id_start in df.index:
        for id_end in df.columns:
            # Skip entries where id_start is equal to id_end
            if id_start != id_end:
                distance = df.loc[id_start, id_end]
                unrolled_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance})

    return unrolled_df

    


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here

    # Convert the list of dictionaries to a DataFrame
    unrolled_df = pd.DataFrame(df)

    # Filter rows for the given reference value
    reference_rows = unrolled_df.loc[unrolled_df['id_start'] == reference_value]

    # Calculate the average distance for the reference value
    average_distance = reference_rows['distance'].mean()

    # Calculate the lower and upper bounds for the threshold (10%)
    lower_bound = average_distance - (0.1 * average_distance)
    upper_bound = average_distance + (0.1 * average_distance)

    # Filter rows within the threshold
    within_threshold = unrolled_df[(unrolled_df['id_start'] != reference_value) & (unrolled_df['distance'] >= lower_bound) & (unrolled_df['distance'] <= upper_bound)]

    # Get unique values from id_start column and sort them
    result_ids = sorted(within_threshold['id_start'].unique())

    return result_ids




def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    result_df = pd.DataFrame(df)

    # Define rate coefficients for each vehicle type
    rate_coefficients = {'moto': 0.8, 'car': 1.2, 'rv': 1.5, 'bus': 2.2, 'truck': 3.6}

    # Calculate toll rates for each vehicle type
    for vehicle_type, rate_coefficient in rate_coefficients.items():
        result_df[vehicle_type] = result_df['distance'] * rate_coefficient

    return result_df

   


def calculate_time_based_toll_rates(df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here

    
   


if __name__ == "__main__":
    try:
        csv_file_path = "C:\Users\ADMIN PC\MapUp-Data-Assessment-F\datasets\dataset-3.csv"
        df = pd.read_csv(csv_file_path)
        distance_matrix = calculate_distance_matrix(df)
        resulting_unrolled_df = unroll_distance_matrix(distance_matrix)
        
        for  row in resulting_unrolled_df:
          reference_value = row['id_start']
          resulting_ids = find_ids_within_ten_percentage_threshold(resulting_unrolled_df, reference_value)
        
        resulting_toll_rates_df = calculate_toll_rate(resulting_unrolled_df)
        print(resulting_toll_rates_df)

        
 
    except Exception as e:
        raise e