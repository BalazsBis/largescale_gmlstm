import pandas as pd
import numpy as np
from pathlib import Path
from numba import njit, prange
from typing import List, Tuple


class global_model():
    """Class to process the Global (initally reduced to the CONUS /continental United States) soil moisture data of ISMN (International Soil Moisture Network)
    
    This class and its methods were taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
    References
    ----------
    .. [#] Dorigo, W., A. Xaver, M. Vreugdenhil, A. Gruber, A. Hegyiová, A. Sanchis-Dufau, D. Zamojski, C. Cordes, W. Wagner, and M. Drusch (2011), The International Soil Moisture Network: A data hosting facility for    
        global in situ soil moisture measurements, Hydrol. Earth Syst. Sci., 15, 1675–1698, https://doi.org/10.5194/hess-15-1675-2011.
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    
    @staticmethod
    def read_attributes(path_data: str) -> pd.DataFrame: 
        """Read the catchments` attributes

        Parameters
        ----------
        path_data : str
            Path to the Attert VWC directory.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the locations` attributes
        """
        # files that contain the attributes
        path_attributes = Path(path_data)
        path_attributes_file = path_attributes / '.csv'
        df_attributes = pd.read_csv(path_attributes_file, sep=',', header=0, dtype={'location_id': str}) # previoulsy ID_sensordepth
        df_attributes.set_index('location_id', inplace=True)

        return df_attributes



    @staticmethod
    def read_data(path_data: str, location_id: str, forcings: List[str]=None)-> pd.DataFrame:
        """Read the locations` timeseries

        Parameters
        ----------
        path_data : str
            Path to the Attert SM measurement locations directory.
        location_id : str
            identifier of the location.
        forcings : List[str]
            Not used, is just to have consistency.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with the locations` timeseries
        """
        path_timeseries = Path(path_data) / 'timeseries' / f'{location_id}.csv'
        # load time series
        df = pd.read_csv(path_timeseries)
        df = df.set_index('date')
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")
        return df


@njit()
def validate_samples(x: np.ndarray, y: np.ndarray, attributes: np.ndarray, seq_length: int, check_NaN:bool=True, 
                     predict_last_n:int=1) -> np.ndarray:
    
    """Checks for invalid samples due to NaN or insufficient sequence length.

    This function was taken from Neural Hydrology [#]_ and adapted for our specific case. 
        
    Parameters
    ----------
    x : np.ndarray
        array of dynamic input;
    y : np.ndarray
        arry of target values;
    attributes : np.ndarray
        array containing the static attributes;
    seq_length : int
        Sequence lengths; one entry per frequency
    check_NaN : bool
        Boolean to specify if Nan should be checked or not
    predict_last_n: int
        Number of values that want to be used to calculate the loss

    Returns
    -------
    flag:np.ndarray 
        Array has a value of 1 for valid samples and a value of 0 for invalid samples.

    References
    ----------
    .. [#] F. Kratzert, M. Gauch, G. Nearing and D. Klotz: NeuralHydrology -- A Python library for Deep Learning
        research in hydrology. Journal of Open Source Software, 7, 4050, doi: 10.21105/joss.04050, 2022 
    """
    # Initialize vector to store the flag. 1 means valid sample for training
    flag = np.ones(x.shape[0])

    for i in prange(x.shape[0]):  # iterate through all samples

        # too early, not enough information
        if i < seq_length - 1:
            flag[i] = 0  
            continue

        if check_NaN:
            # any NaN in the dynamic inputs makes the sample invalid
            x_sample = x[i-seq_length+1 : i+1, :]
            if np.any(np.isnan(x_sample)):
                flag[i] = 0
                continue

        if check_NaN:
            # all-NaN in the targets makes the sample invalid
            y_sample = y[i-predict_last_n+1 : i+1]
            if np.all(np.isnan(y_sample)):
                flag[i] = 0
                continue

        # any NaN in the static features makes the sample invalid
        if attributes is not None and check_NaN:
            if np.any(np.isnan(attributes)):
                flag[i] = 0

    return flag

def manual_train_test_split(segments, test_size=0.4, random_state=None):
    '''Splits segments randomly into training and testing subsets'''
    
    # Ensure random reproducibility
    if random_state is not None:
        random.seed(random_state)
    
    # Shuffle the segments
    random.shuffle(segments)
    
    # Calculate the split index
    test_index = int(len(segments) * test_size)
    
    # Split into training and test sets
    test_segments = segments[:test_index]
    train_segments = segments[test_index:]
    
    return train_segments, test_segments