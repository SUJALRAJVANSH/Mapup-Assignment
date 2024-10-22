from typing import Dict, List
import re
import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements without using slicing or the reverse function.
    Args:
        lst (List[int]): The list to reverse in chunks.
        n (int): Number of elements in each chunk.
    Returns:
        List[int]: The modified list with every group of n elements reversed.
    """
    result = []
    i = 0
    while i < len(lst):
        chunk = []
        for j in range(i, min(i + n, len(lst))):
            chunk.insert(0, lst[j])  # Insert elements in reverse order
        result.extend(chunk)
        i += n
    return result


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    Args:
        lst (List[str]): List of strings to group by length.
    Returns:
        Dict[int, List[str]]: Dictionary where keys are lengths, and values are lists of strings of that length.
    """
    result = {}
    for string in lst:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)
    return dict(sorted(result.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    Args:
        nested_dict (Dict): The dictionary object to flatten.
        sep (str): The separator to use between parent and child keys (defaults to '.').
    Returns:
        Dict: A flattened dictionary.
    """
    def recurse(d, parent_key=''):
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(recurse(v, new_key).items())
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    items.extend(recurse({f'{new_key}[{i}]': item}).items())
            else:
                items.append((new_key, v))
        return dict(items)

    return recurse(nested_dict)


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    Args:
        nums (List[int]): List of integers (may contain duplicates).
    Returns:
        List[List[int]]: List of unique permutations.
    """
    def backtrack(path, used):
        if len(path) == len(nums):
            res.append(path[:])
            return
        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False

    nums.sort()
    res = []
    backtrack([], [False] * len(nums))
    return res


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    Args:
        text (str): A string containing the dates in various formats.
    Returns:
        List[str]: A list of valid dates in the formats specified.
    """
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',   # dd-mm-yyyy
        r'\b\d{2}/\d{2}/\d{4}\b',   # mm/dd/yyyy
        r'\b\d{4}\.\d{2}\.\d{2}\b'  # yyyy.mm.dd
    ]
    dates = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        dates.extend(matches)
    return dates


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    Args:
        polyline_str (str): The encoded polyline string.
    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    import polyline  # polyline module is required to decode
    coordinates = polyline.decode(polyline_str)
    
    # Haversine formula for distance calculation
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000  # Radius of the Earth in meters
        phi1 = radians(lat1)
        phi2 = radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)
        a = sin(delta_phi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(delta_lambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return R * c  # Output in meters
    
    data = []
    for i, (lat, lon) in enumerate(coordinates):
        distance = 0 if i == 0 else haversine(coordinates[i - 1][0], coordinates[i - 1][1], lat, lon)
        data.append([lat, lon, distance])
    
    return pd.DataFrame(data, columns=["latitude", "longitude", "distance"])


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then replace each element with the sum of all elements in the same row
    and column (in the rotated matrix), excluding itself.
    Args:
        matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    Returns:
        List[List[int]]: A new 2D list representing the transformed matrix.
    """
    n = len(matrix)
    rotated = [[matrix[n - j - 1][i] for j in range(n)] for i in range(n)]
    
    final_matrix = []
    for i in range(n):
        row_sum = sum(rotated[i])
        new_row = []
        for j in range(n):
            col_sum = sum(rotated[k][j] for k in range(n))
            new_row.append(row_sum + col_sum - 2 * rotated[i][j])
        final_matrix.append(new_row)
    return final_matrix


def time_check(df) -> pd.Series:
    """
    Verifies if the timestamps for each unique (id, id_2) pair cover a full 24-hour period and 7-day week.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.Series: A boolean series with a multi-index (id, id_2).
    """
    # Assuming df contains the following columns: 'id', 'id_2', 'startDay', 'startTime', 'endDay', 'endTime'
    df['start'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'])
    df['end'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'])
    
    def check_complete(group):
        # Group should span full 7 days and cover 24 hours each day
        days_covered = group['startDay'].nunique() == 7
        hours_covered = (group['end'] - group['start']).sum() >= pd.Timedelta('7 days')
        return days_covered and hours_covered
    
    return df.groupby(['id', 'id_2']).apply(check_complete)
