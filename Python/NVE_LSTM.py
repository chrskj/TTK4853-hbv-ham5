#!/usr/bin/env python
# coding: utf-8

# # Long Short-Term Memory (LSTM) for rainfall-runoff modelling
# 
# Recently, Kratzert et al. (2018a, 2018b) have shown the potential of LSTMs for rainfall-runoff modelling. Here, I'll show some example for setting up and training such a model. For single catchment calibration (as done in this notebook) no GPU is required, and everything should run fine on a standard CPU (also rather slow, compared to a GPU).
# 
# In this example, we use the NVE data set (provided by Elhadi Mohsen Hassan Abdalla) that provides us with approx. 15 years of daily meteorological forcings and discharge observations from 100 catchments across Norway. As input to our model we use daily precipitation and temperature. 
# 
# - Kratzert, F., Klotz, D., Brenner, C., Schulz, K., and Herrnegger, M.: Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks, Hydrol. Earth Syst. Sci., 22, 6005-6022, https://doi.org/10.5194/hess-22-6005-2018, 2018a. 
# 
# - Kratzert F., Klotz D., Herrnegger M., Hochreiter S.: A glimpse into the Unobserved: Runoff simulation for ungauged catchments with LSTMs, Workshop on Modeling and Decision-Making in the Spatiotemporal Domain, 32nd Conference on Neural Information Processing Systems (NeuRIPS 2018), Montréal, Canada. [https://openreview.net/forum?id=Bylhm72oKX](https://openreview.net/forum?id=Bylhm72oKX), 2018b.
# 
# - A. Newman; K. Sampson; M. P. Clark; A. Bock; R. J. Viger; D. Blodgett, 2014. A large-sample watershed-scale hydrometeorological dataset for the contiguous USA. Boulder, CO: UCAR/NCAR. https://dx.doi.org/10.5065/D6MW2F4D
# 
# This is notebook is based on: https://github.com/kratzert/pangeo_lstm_example.
# 
# Date: 25.03.2020<br/>

# In[106]:


# Imports
from pathlib import Path
from typing import Tuple, List

import gcsfs
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import glob

# Globals
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # This line checks if GPU is available


# ## Data loading
# ### NVE specific data loader functions
# 
# Next define a function to load the meteorological forcings and the discharge for any specific catchment from the NVE data set.

# In[107]:


def load_forcing_and_discharge(catchment: int) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data of a specific catchment.

    :param catchment: number (id)
    
    :return: pd.DataFrame containing the meteorological forcing data.
    """
    path = '../Input files (.txt)'
    all_files = glob.glob(path + "/*.txt")

    file_exist = False
    
    # Loop through files and find correct catchment
    for file_path in all_files:
        # Name is formatted `./Input files (.txt)/nve_inp_XX.txt`
        number = int(file_path.split('_')[-1].split('.')[0])
        
        if number == catchment:
            file_exist = True
            df = pd.read_csv(file_path, encoding='cp1252', skiprows=[0], delimiter=r"\s+", parse_dates=[['dd.mm.yyyy',  'hh:mm:ss']])
            df = df.rename(columns={"dd.mm.yyyy_hh:mm:ss": "timestamp"})
    
    # Return None if catchment does not exist
    if file_exist == False:
        print("Catchment does not exist")
        return None
    else:
        return df


# In[108]:


# Display Data frame for one catchment
#display(load_forcing_and_discharge(file_nr))


# ## Import training data

# In[109]:


def load_training_data(catchment: int) -> Tuple[pd.DataFrame, int]:
    """Load the meteorological forcing data and other characteristics of a specific catchment.

    :param catchment: number (id)
    
    :return: pd.DataFrame containing the meteorological forcing and other characteristics data.
    """
    path = '../Training_data'
    all_files = glob.glob(path + "/*.csv")

    file_exist = False
    
    # Loop through files and find correct catchment
    for file_path in all_files:
        # Name is formatted `./Input files (.txt)/nve_inp_XX.txt`
        number = int(file_path.split('_')[-1].split('.')[0])
        
        if number == catchment:
            file_exist = True
            df = pd.read_csv(file_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            # Delay by one day
            df["Residual_in"] = np.roll(df["Residual"], 1)
            
    # Return None if catchment does not exist
    if file_exist == False:
        print("Catchment does not exist")
        return None
    else:
        return df


# In[110]:


# Display training data frame for one catchment
#display(load_training_data(file_nr))


# ### Data reshaping for LSTM training
# Next we need another utility function to reshape the data into an appropriate format for training LSTMs. These recurrent neural networks expect sequential input of the shape `(sequence length, number of features)`. We train our network to predict a single day of discharge from *n* days of precendent meteorological observations. For example, lets assume that _n_ = 365, then a single training sample should be of shape `(365, number of features)`, and since we use 2 input features the shape is `(365, 2)`. Here the features are temperature and rain fall.
# 
# However, when loaded from the files the entire data is stored in a matrix, where the number of rows correspond to the total number of days in the training set and the number of columns to the features. Thus we need to slide over this matrix and cut out small samples appropriate to our LSTM setting. To speed things up, we make use of the awesome Numba library here (the little @njit decorator JIT-compiles this function and dramatically increases the speed).

# In[111]:


@njit
def reshape_data(x: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reshape matrix data into sample shape for LSTM training.

    :param x: Matrix containing input features column wise and time steps row wise
    :param y: Matrix containing the output feature.
    :param seq_length: Length of look back days for one day of prediction
    
    :return: Two np.ndarrays, the first of shape (samples, length of sequence,
        number of features), containing the input data for the LSTM. The second
        of shape (samples, 1) containing the expected output for each input
        sample.
    """
    num_samples, num_features = x.shape

    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features))
    y_new = np.zeros((num_samples - seq_length + 1, 1))

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new


# ### PyTorch data set
# Now we wrap everything into a PyTorch Dataset. These are specific classes that can be used by PyTorchs DataLoader class for generating mini-batches (and do this in parallel in multiple threads). Such a data set class has to inherit from the PyTorch Dataset class and three functions have to be implemented
# 
# 1. __init__(): The object initializing function.
# 2. __len__(): This function has to return the number of samples in the data set.
# 3. __getitem__(i): A function that returns sample `i` of the data set (sample + target value).

# In[112]:


class NVE_TXT(Dataset):
    """Torch Dataset for basic use of data from the NVE data set.

    This data set provides meteorological observations and discharge of a given
    catchment from the NVE data set.
    """

    def __init__(self, catchment: int, seq_length: int=365,period: str=None,
                 dates: List=None, means: pd.Series=None, stds: pd.Series=None):
        """Initialize Dataset containing the data of a single catchment.

        :param catchment: number (id) indicating a given NVE catchment 
        :param seq_length: (optional) Length of the time window of
            meteorological input provided for one time step of prediction.
        :param period: (optional) One of ['train', 'eval']. None loads the 
            entire time series.
        :param dates: (optional) List of pd.DateTimes of the start and end date 
            of the discharge period that is used.
        :param means: (optional) Means of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_means() on the data set.
        :param stds: (optional) Stds of input and output features derived from
            the training period. Has to be provided for 'eval' period. Can be
            retrieved if calling .get_stds() on the data set.
        """
        self.catchment = catchment
        self.seq_length = seq_length
        self.period = period
        self.dates = dates
        self.means = means
        self.stds = stds
            
        # load data into memory
        self.x, self.y = self._load_data()

        # store number of samples as class attribute
        self.num_samples = self.x.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]

    def _load_data(self):
        """Load input and output data from text files."""
        #df = load_forcing_and_discharge(self.catchment)
        df = load_training_data(self.catchment)
        if self.dates is not None:
            # Truncate data after start and end date
            df = df[df['timestamp'] >= self.dates[0]]
            df = df[df['timestamp'] < self.dates[1]]
        
        # if training period store means and stds
        if self.period == 'train':
            self.means = df.mean()
            self.stds = df.std()         
        
        # extract input and output features from DataFrame
        x = np.array([df['mm1'].values,
                      df['mm2'].values,
                      df['mm3'].values,
                      df['mm4'].values,
                      df['mm5'].values,
                      df['mm6'].values,
                      df['mm7'].values,
                      df['mm8'].values,
                      df['mm9'].values,
                      df['mm10'].values,
                      df['grC1'].values,
                      df['grC2'].values,
                      df['grC3'].values,
                      df['grC4'].values,
                      df['grC5'].values,
                      df['grC6'].values,
                      df['grC7'].values,
                      df['grC8'].values,
                      df['grC9'].values,
                      df['grC10'].values,
                      df['Residual_in'].values,
                      df['EPOT'].values]).T
        y = np.array([df['Residual'].values]).T
        
        # normalize data, reshape for LSTM training and remove invalid samples
        x = self._local_normalization(x, variable='inputs')
        x, y = reshape_data(x, y, self.seq_length)                      

        if self.period == "train":            
            # normalize discharge
            y = self._local_normalization(y, variable='output')

        # convert arrays to torch tensors
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return x, y

    def _local_normalization(self, feature: np.ndarray, variable: str) ->             np.ndarray:
        """Normalize input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['mm1'],
                              self.means['mm2'],
                              self.means['mm3'],
                              self.means['mm4'],
                              self.means['mm5'],
                              self.means['mm6'],
                              self.means['mm7'],
                              self.means['mm8'],
                              self.means['mm9'],
                              self.means['mm10'],
                              self.means['grC1'],
                              self.means['grC2'],
                              self.means['grC3'],
                              self.means['grC4'],
                              self.means['grC5'],
                              self.means['grC6'],
                              self.means['grC7'],
                              self.means['grC8'],
                              self.means['grC9'],
                              self.means['grC10'],
                              self.means['Residual_in'],
                              self.means['EPOT']])
            stds = np.array([self.stds['mm1'],
                             self.stds['mm2'],
                             self.stds['mm3'],
                             self.stds['mm4'],
                             self.stds['mm5'],
                             self.stds['mm6'],
                             self.stds['mm7'],
                             self.stds['mm8'],
                             self.stds['mm9'],
                             self.stds['mm10'],
                             self.stds['grC1'],
                             self.stds['grC2'],
                             self.stds['grC3'],
                             self.stds['grC4'],
                             self.stds['grC5'],
                             self.stds['grC6'],
                             self.stds['grC7'],
                             self.stds['grC8'],
                             self.stds['grC9'],
                             self.stds['grC10'],
                             self.stds['Residual_in'],
                             self.stds['EPOT']])
            feature = (feature - means) / stds
        elif variable == 'output':
            feature = ((feature - self.means["Residual"]) /
                       self.stds["Residual"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature
    
    def local_rescale(self, feature: np.ndarray, variable: str) ->             np.ndarray:
        """Rescale input/output features with local mean/std.

        :param feature: Numpy array containing the feature(s) as matrix.
        :param variable: Either 'inputs' or 'output' showing which feature will
            be normalized
        :return: array containing the normalized feature
        """
        if variable == 'inputs':
            means = np.array([self.means['mm1'],
                              self.means['mm2'],
                              self.means['mm3'],
                              self.means['mm4'],
                              self.means['mm5'],
                              self.means['mm6'],
                              self.means['mm7'],
                              self.means['mm8'],
                              self.means['mm9'],
                              self.means['mm10'],
                              self.means['grC1'],
                              self.means['grC2'],
                              self.means['grC3'],
                              self.means['grC4'],
                              self.means['grC5'],
                              self.means['grC6'],
                              self.means['grC7'],
                              self.means['grC8'],
                              self.means['grC9'],
                              self.means['grC10'],
                              self.means['Residual_in'],
                              self.means['EPOT']])
            stds = np.array([self.stds['mm1'],
                             self.stds['mm2'],
                             self.stds['mm3'],
                             self.stds['mm4'],
                             self.stds['mm5'],
                             self.stds['mm6'],
                             self.stds['mm7'],
                             self.stds['mm8'],
                             self.stds['mm9'],
                             self.stds['mm10'],
                             self.stds['grC1'],
                             self.stds['grC2'],
                             self.stds['grC3'],
                             self.stds['grC4'],
                             self.stds['grC5'],
                             self.stds['grC6'],
                             self.stds['grC7'],
                             self.stds['grC8'],
                             self.stds['grC9'],
                             self.stds['grC10'],
                             self.stds['Residual_in'],
                             self.stds['EPOT']])
            feature = feature * stds + means
        elif variable == 'output':
            feature = (feature * self.stds["Residual"] +
                       self.means["Residual"])
        else:
            raise RuntimeError(f"Unknown variable type {variable}")

        return feature

    def get_means(self):
        return self.means

    def get_stds(self):
        return self.stds


# ## Build LSTM model
# 
# Here we implement a single layer LSTM with optional dropout in the final fully connected layer. Using PyTorch, we need to inherit from the `nn.Module` class and implement the `__init__()`, as well as a `forward()` function. To make things easy, we make use of the standard LSTM layer included in the PyTorch library.
# 
# The forward function implements the entire forward pass through the network, while the backward pass, used for updating the weights during training, is automatically derived from PyTorch autograd functionality.

# In[113]:


class Model(nn.Module):
    """Implementation of a single layer LSTM network"""
    
    def __init__(self, hidden_size: int, dropout_rate: float=0.0):
        """Initialize model
        
        :param hidden_size: Number of hidden units/LSTM cells
        :param dropout_rate: Dropout rate of the last fully connected
            layer. Default 0.0
        """
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # create required layer
        self.lstm = nn.LSTM(input_size=22, hidden_size=self.hidden_size, 
                            num_layers=1, bias=True, batch_first=True)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Network.
        
        :param x: Tensor of shape [batch size, seq length, num features]
            containing the input data for the LSTM network.
        
        :return: Tensor containing the network predictions
        """
        output, (h_n, c_n) = self.lstm(x)
        
        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred


# ## Train utilities
# 
# Next we are implementing two functions for training and evaluating the model.
# 
# - `train_epoch()`: This function iterates one time over the entire training data set (called one epoch) and updates the weights of the network to minimize the loss function (we use the mean squared error here).
# - `eval_model()`: To this function evaluates a data set and returns the predictions, as well as observations.
# 
# Furthermore we implement a function `calc_nse()` to calculate the Nash-Sutcliffe-Efficiency for our model

# In[114]:


def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorchs optimizer classes.
    :param loader: A PyTorch DataLoader, providing the trainings
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    # set model to train mode (important for dropout)
    model.train()
    pbar = tqdm.tqdm_notebook(loader)
    pbar.set_description(f"Epoch {epoch}")
    # request mini-batch of data from the loader
    for xs, ys in pbar:
        # delete previously stored gradients from the model
        optimizer.zero_grad()
        # push data to GPU (if available)
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        # get model predictions
        y_hat = model(xs)
        # calculate loss
        loss = loss_func(y_hat, ys)
        # calculate gradients
        loss.backward()
        # update the weights
        optimizer.step()
        # write current loss in the progress bar
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        
def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.
    
    :return: Two torch Tensors, containing the observations and 
        model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    # in inference mode, we don't need to store intermediate steps for
    # backprob
    with torch.no_grad():
        # request mini-batch of data from the loader
        for xs, ys in loader:
            # push data to GPU (if available)
            xs = xs.to(DEVICE)
            # get model predictions
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
            
    return torch.cat(obs), torch.cat(preds)
        
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

def load_sim_obs_data(catchment: int, dates: List=None):
        """Initialize Dataset containing the data of a single catchment.

        :param catchment: number (id) indicating a given NVE catchment 
        :param dates: (optional) List of pd.DateTimes of the start and end date 
            of the discharge period that is used.
        """
        df = load_training_data(catchment)
        if dates is not None:
            # Truncate data after start and end date
            df = df[df['timestamp'] >= dates[0]]
            df = df[df['timestamp'] < dates[1]]


        sim = np.array([df['Q_sim'].values]).T
        obs = np.array([df['Q_obs'].values]).T

        _, obs_out = reshape_data(sim, obs, 365)  
        _, sim_out = reshape_data(obs, sim, 365)  

        return sim_out, obs_out

# ## Prepare everything for training
# 
# Now that we have everything needed, we have to do the following steps:
# 
# - Specify a training, as well independent validation and test periods. For each of the three create a PyTorch data set and with these data sets PyTorch DataLoader. These DataLoader put single samples together to mini-batches that we use to train the network.
# - Initialize the model
# - Create a optimizer
# - Define the loss function

# In[115]:

file_nrs = [1, 6, 9, 14, 18, 25, 30, 38, 44, 65, 66, 88, 92, 94, 97, 99] 
for file_nr in file_nrs:
    print("#"*20)
    print(f"Catchment nr: {file_nr}")
    print("#"*20)
    catchment = file_nr # can be changed to any catchment id contained in the NVE data set
    hidden_size = 10 # Number of LSTM cells
    dropout_rate = 0.0 # Dropout rate of the final fully connected Layer [0.0, 1.0]
    learning_rate = 10*1e-3 # Learning rate used to update the weights
    sequence_length = 365 # Length of the meteorological record provided to the network

    ##############
    # Data set up#
    ##############

    # Training data
    start_date = pd.to_datetime("2000-09-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2007-08-31", format="%Y-%m-%d")
    ds_train = NVE_TXT(catchment, seq_length=sequence_length, period="train", dates=[start_date, end_date])
    tr_loader = DataLoader(ds_train, batch_size=256, shuffle=True)
    print(f"Training data length: {len(ds_train)}")

    # Validation data. We use the feature means/stds of the training period for normalization
    means = ds_train.get_means()
    stds = ds_train.get_stds()
    start_date = pd.to_datetime("2007-09-01", format="%Y-%m-%d")
    end_date = pd.to_datetime("2009-08-31", format="%Y-%m-%d")
    ds_val = NVE_TXT(catchment, seq_length=sequence_length, period="eval", dates=[start_date, end_date],
                         means=means, stds=stds)
    val_loader = DataLoader(ds_val, batch_size=2048, shuffle=False)
    print(f"Validation data length: {len(ds_val)}")

    # Test data. We use the feature means/stds of the training period for normalization
    start_date_test = pd.to_datetime("2007-09-01", format="%Y-%m-%d")
    end_date_test = pd.to_datetime("2015-08-31", format="%Y-%m-%d")
    ds_test = NVE_TXT(catchment, seq_length=sequence_length, period="eval", dates=[start_date_test, end_date_test],
                         means=means, stds=stds)
    test_loader = DataLoader(ds_test, batch_size=2048, shuffle=False)
    print(f"Test data length: {len(ds_test)}")

    #########################
    # Model, Optimizer, Loss#
    #########################

    # Here we create our model, feel free 
    model = Model(hidden_size=hidden_size, dropout_rate=dropout_rate).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss()


    # ## Train the model
    # Now we gonna train the model for some number of epochs. After each epoch we evaluate the model on the validation period and print out the NSE.

    # In[ ]:


    n_epochs = 10 # Number of training epochs

    for i in range(n_epochs):
        train_epoch(model, optimizer, tr_loader, loss_func, i+1)
        obs, preds = eval_model(model, val_loader)
        preds = ds_val.local_rescale(preds.numpy(), variable='output')
        nse = calc_nse(obs.numpy(), preds)
        tqdm.tqdm.write(f"Validation NSE: {nse:.3f}")


    # ## Evaluate independent test set
    # Finally, we can can evaluate our model on the unseen test data, calculate the NSE and plot some observations vs predictions

    # ### Residual NSE plot

    # In[ ]:


    # Evaluate on test set
    obs, preds = eval_model(model, test_loader)
    preds = ds_val.local_rescale(preds.numpy(), variable='output')
    obs = obs.numpy()
    nse = calc_nse(obs, preds)
    max_abs_error = max(abs(preds-obs))

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 4))

    #start_date = ds_test.dates[0]
    #end_date = ds_test.dates[1] + pd.DateOffset(days=1)
    #date_range = pd.date_range(start_date, end_date)

    plt.figure(figsize=(20, 16))
    ax.plot(obs, label="observation")
    ax.plot(preds, label="prediction")
    ax.legend()

    ax.set_title(f"Catchment {catchment} - Residual NSE: {nse:.3f} - Max abs error: {max_abs_error}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Days")
    _ = ax.set_ylabel("Residuals (m3/s)")

    fig.savefig(f"LSTM Plots/LSTM_Residuals_&_BB_test_NSE_{file_nr}.png")
    
    d = {"predicted_residual": list(preds), "observed_residual": list(obs)}
    output_df = pd.DataFrame.from_dict(d)

    # In[ ]:


 



    # ### Simulation model NSE plot

    # In[ ]:


    # Evaluate on test set
    Q_sim, Q_obs = load_sim_obs_data(catchment, dates=[start_date_test, end_date_test])

    Q_sim_sum = Q_sim

    #obs = obs.numpy()
    nse = calc_nse(Q_obs, Q_sim_sum)
    max_abs_error = max(abs(Q_sim_sum-Q_obs))

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 4))

    #start_date = ds_test.dates[0]
    #end_date = ds_test.dates[1] + pd.DateOffset(days=1)
    #date_range = pd.date_range(start_date, end_date)

    plt.figure(figsize=(20, 16))
    ax.plot(Q_obs, label="observation")
    ax.plot(Q_sim_sum, label="prediction")
    ax.legend()

    ax.set_title(f"Catchment {catchment} - Simulated NSE: {nse:.3f} - Max abs error: {max_abs_error}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Days")
    _ = ax.set_ylabel("Runoff (m3/s)")

    fig.savefig(f"LSTM Plots/LSTM_Simulated_test_NSE_{file_nr}.png")
    
    output_df["hbv_simulated"] = list(Q_sim_sum)
    output_df["hbv_observed"] = list(Q_obs)

    # ### Total NSE plot

    # In[ ]:


    # Evaluate on test set
    obs, preds = eval_model(model, test_loader)
    preds = ds_val.local_rescale(preds.numpy(), variable='output')

    Q_sim, Q_obs = load_sim_obs_data(catchment, dates=[start_date_test, end_date_test])

    Q_sim_sum = Q_sim -preds

    #obs = obs.numpy()
    nse = calc_nse(Q_obs, Q_sim_sum)
    max_abs_error = max(abs(Q_sim_sum-Q_obs))

    # Plot results
    fig, ax = plt.subplots(figsize=(12, 4))

    #start_date = ds_test.dates[0]
    #end_date = ds_test.dates[1] + pd.DateOffset(days=1)
    #date_range = pd.date_range(start_date, end_date)

    plt.figure(figsize=(20, 16))
    ax.plot(Q_obs, label="observation")
    ax.plot(Q_sim_sum, label="prediction")
    ax.legend()

    ax.set_title(f"Catchment {catchment} - Simulated+BB NSE: {nse:.3f} - Max abs error: {max_abs_error}")
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Days")
    _ = ax.set_ylabel("Runoff (m3/s)")

    fig.savefig(f"LSTM Plots/LSTM_Simulated_&_BB_test_NSE_{file_nr}.png")

    output_df["hbv_BB_simulated"] = list(Q_sim_sum)
    output_df["hbv_BB_observed"] = list(Q_obs)
    # In[ ]:
    
    # Save to csv file
    output_df.to_csv(f"lstm_no_lookback_{file_nr}.csv")



