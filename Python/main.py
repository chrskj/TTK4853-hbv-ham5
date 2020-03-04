import numpy as np
import pandas as pd
import glob


# From https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb
def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    # only consider time steps, where observations are available
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    # check for NaNs in observations
    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val

# Read all input txt files
path = '../Input files (.txt)'
all_files = glob.glob(path + "/*.txt")

df_dict = {}
for file_path in all_files:
    #print(f'Reading file: {file_path}')
    # Name is formatted `./Input files (.txt)/nve_inp_XX.txt`
    number = int(file_path.split('_')[-1].split('.')[0])

    df = pd.read_csv(file_path, encoding='cp1252', skiprows=[0], delimiter=r"\s+", parse_dates=[['dd.mm.yyyy',  'hh:mm:ss']])
    #df = df.rename(columns={"dd.mm.yyyy_hh:mm:ss": "timestamp"})
    df_dict[number] = df



    