import pandas as pd

import numpy as np

def generate_car_matrix(df)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values, 
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    df= df.pivot(index='id_1', columns='id_2', values='car').fillna(0) #values from id_2 as columns ,values from id_1 as index and dataframe  have values from car column
   
    df.values[np.arange(df.shape[0]), np.arange(df.shape[0])] = 0 # setting diagonal values to zero
   
    return df


def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    df['car_type'] = pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)
  
    count_type = df['car_type'].value_counts().to_dict()

    return dict(sorted(count_type.items()))


def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here

    df['bus'] = pd.to_numeric(df['bus'], errors='coerce')
    bus_mean = df['bus'].mean()
    bus_indexes = sorted(df.index[df['bus'] > 2 * bus_mean])
    return list(bus_indexes)


def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    df['truck'] = pd.to_numeric(df['truck'], errors='coerce')
    route_means = df.groupby('route')['truck'].mean()
    filtered_routes = sorted(route_means[route_means > 7])
    

    return list((filtered_routes))


def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    matrix1 = generate_car_matrix(matrix)
    matrix2 = matrix1.applymap(lambda x: x * 0.75 if x > 20 else x * 1.25)
    matrix2 = matrix2.round(1)

    return matrix2


def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()

def main():

    df = pd.read_csv('C:\Users\ADMIN PC\MapUp-Data-Assessment-F\datasets\dataset-1.csv')

    car_matrix = generate_car_matrix(df)
    print("Car Matrix:")
    print(car_matrix)

    type_count = get_type_count(df)
    print("\nCar Type Count:")
    print(type_count)

    bus_indexes = get_bus_indexes(df)
    print("\nBus Indexes:")
    print(bus_indexes)

    filtered_routes = filter_routes(df)
    print("\nFiltered Routes:")
    print(filtered_routes)

    multiplied_matrix = multiply_matrix(df)
    print("\nMultiplied Matrix:")
    print(multiplied_matrix)


if __name__ == "__main__":
    try:
      main()
    except Exception as e:
        raise e

