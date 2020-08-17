# libraries
# region
#################################################
import glob
import timeit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.metrics import silhouette_score
import seaborn as sns
import random
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

starttime = timeit.default_timer()
sns.set()
from mpl_toolkits.mplot3d import Axes3D
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

pd.set_option("display.max_rows", 500)
# endregion

############## load datasets ################

# Get all the file names
mat_files_name = glob.glob("data/Larsen*")
num_mat_files = len(mat_files_name)

# Load all the files
mat_files = [loadmat(x) for x in mat_files_name]

# create dataframes for variables

features = ["Euu_k", "Evv_k", "Eww_k", "k"]
col_names = ["u", "v", "w", "k"]
dfs = []
for num, feat in enumerate(features):
    data = [mat_files[i][feat].reshape(-1).transpose() for i in range(num_mat_files)]
    combined = np.stack(data)
    cols = [str(col_names[num]) + str(i) for i in range(2211)]
    df = pd.DataFrame(combined, index=mat_files_name, columns=cols)
    df["item"] = df.index.str.split("\\").str[1]
    df = df.set_index("item")
    dfs.append(df)

Euu_df = dfs[0]
Evv_df = dfs[1]
Eww_df = dfs[2]
k_df = dfs[3]

# checks
Euu_df.head()
Euu_df.shape

# save raw files
Euu_df.to_csv("Euu_df.csv")
Evv_df.to_csv("Evv_df.csv")
Eww_df.to_csv("Eww_df.csv")
k_df.to_csv("k_df.csv")


# combine Euu, Evv and Eww

readings_df = Euu_df.join(Evv_df).join(Eww_df)

# checks
readings_df.head()
readings_df.shape

#########################################

############## functions ################


def draw_spectra(item, Euu_df=Euu_df, Evv_df=Evv_df, Eww_df=Eww_df, k_df=k_df):
    """This code draws the log wave spectra graph
    with three gradient lines over a set range"""
    # draw graph
    f, ax = plt.subplots(figsize=(6, 6))
    plt.loglog(k_df.loc[item], Euu_df.loc[item], label="Euu_k")
    plt.loglog(k_df.loc[item], Evv_df.loc[item], label="Evv_k")
    plt.loglog(k_df.loc[item], Eww_df.loc[item], label="Eww_k")
    # graph settings
    plt.ylim((0.00000000001, 0.001))
    plt.title(f"{idx} Spectra")
    plt.margins(0.25, 0.75)
    # gridlines
    gridline3y = np.exp((-5 / 3) * (np.log(k_df.loc[item])) - 9)
    gridline4y = np.exp((-5 / 3) * (np.log(k_df.loc[item])) - 13)
    gridline5y = np.exp((-5 / 3) * (np.log(k_df.loc[item])) - 17)
    plt.loglog(df[x_k_key][idx], gridline3y, c="gray", linestyle="dashed")
    plt.loglog(df[x_k_key][idx], gridline4y, c="gray", linestyle="dashed")
    plt.loglog(df[x_k_key][idx], gridline5y, c="gray", linestyle="dashed")
    plt.legend()


def get_part_gradient(df, start, end, k_df=k_df):
    """This function takes a starting k value and a end k value,
    finds the nearest index in the k dataset and finds the gradient
    and intersect between those points for the defined df."""
    gradient = np.zeros((8628,))
    intersect = np.zeros((8628,))
    for idx, name in enumerate(df.index):
        first = k_df.loc[name].searchsorted(start)
        last = k_df.loc[name].searchsorted(end)
        m, c = np.polyfit(
            np.log(k_df.loc[name][first:last]), np.log(df.loc[name][first:last]), 1
        )
        gradient[idx] += m
        intersect[idx] += c
    return gradient, intersect


def mse(col, grad, intercept, idx):
    m = df[grad][idx]
    c = df[intercept][idx]
    logk = np.log(df["k"][idx])
    y_fit = np.exp(m * logk + c)
    return mean_squared_error(df[col][idx], y_fit)


def get_umean(df):
    """This returns the mean per reading. To only be used with U 
    as V and W rotated to zero.(Time domain function)."""
    return df.mean(axis=1)


def get_std(df):
    """Returns std per reading for the df."""
    log = np.log(df)
    return log.std(axis=1)


def get_first_std(Euu_df=Euu_df, Evv_df=Evv_df, Eww_df=Eww_df):
    """Returns the std of the first points for each reading"""
    first_std = np.zeros((8628,))
    for index, item in enumerate(df.index):
        u = np.log(Euu_df.loc[item][0])
        v = np.log(Evv_df.loc[item][0])
        w = np.log(Eww_df.loc[item][0])
        entry = np.std(np.array([u, v, w]))
        first_std[index] += entry
    return first_std


def plot_ewm(item, window, Euu_df=Euu_df, Evv_df=Evv_df, Eww_df=Eww_df, k_df=k_df):
    u_rolling = Euu_df.loc[item].ewm(span=window).mean()
    v_rolling = Evv_df.loc[item].ewm(span=window).mean()
    w_rolling = Eww_df.loc[item].ewm(span=window).mean()
    k = k_df.loc[item]

    # plot data
    plt.loglog(k, u_rolling, label="Euu")
    plt.loglog(k, v_rolling, label="Evv")
    plt.loglog(k, w_rolling, label="Eww")

    # plot gridlines
    x = np.linspace(10, 1000, 100)
    gridline3y = np.exp((-5 / 3) * (np.log(x)) - 1)
    gridline4y = np.exp((-5 / 3) * (np.log(x)) - 5)
    gridline5y = np.exp((-5 / 3) * (np.log(x)) - 9)
    plt.loglog(x, gridline3y, c="gray", linestyle="dashed")
    plt.loglog(x, gridline4y, c="gray", linestyle="dashed")
    plt.loglog(x, gridline5y, c="gray", linestyle="dashed")
    plt.legend()
    plt.title(f"{item} Exponentially Smoothed")


def plot_rolling(item, window, Euu_df=Euu_df, Evv_df=Evv_df, Eww_df=Eww_df, k_df=k_df):
    u_rolling = Euu_df.loc[item].rolling(window=window).mean()
    v_rolling = Evv_df.loc[item].rolling(window=window).mean()
    w_rolling = Eww_df.loc[item].rolling(window=window).mean()
    k = k_df.loc[item]

    start = window - 1

    # plot data
    plt.loglog(k[start:], u_rolling[start:], label="Euu")
    plt.loglog(k[start:], v_rolling[start:], label="Evv")
    plt.loglog(k[start:], w_rolling[start:], label="Eww")

    # plot gridlines
    x = np.linspace(10, 1000, 100)
    gridline3y = np.exp((-5 / 3) * (np.log(x)) - 1)
    gridline4y = np.exp((-5 / 3) * (np.log(x)) - 5)
    gridline5y = np.exp((-5 / 3) * (np.log(x)) - 9)
    plt.loglog(x, gridline3y, c="gray", linestyle="dashed")
    plt.loglog(x, gridline4y, c="gray", linestyle="dashed")
    plt.loglog(x, gridline5y, c="gray", linestyle="dashed")
    plt.legend()
    plt.title(f"{item} Rolling")


def start_dec_lim(k0):
    ##denotes starting decade for reading
    if k0 < 0.01:
        return 0.01
    if 0.01 < k0 < 0.1:
        return 0.1
    if 0.1 < k0 < 1:
        return 1
    if 1 < k0 < 10:
        return 10


########################## get features #############################

######### get exponential smoothing #########
"""This creates exponentially smoothed data over window specified.
    Note window = 150 is 1 minute and 750 is 5 minutes"""
window = 750
# create smoothing with chosen window
u_rolling = np.log(Euu_df.ewm(span=window, axis=1).mean())
v_rolling = np.log(Evv_df.ewm(span=window, axis=1).mean())
w_rolling = np.log(Eww_df.ewm(span=window, axis=1).mean())

# create log of k
logk = pd.DataFrame(np.log(k_df).values, index=k_df.index, columns=k_df.columns)

# calculate gradient and intersect and separate into columns
roll_window = 750
# create smoothing with chosen window
u_rollinge = np.log(Euu_df.ewm(span=roll_window, axis=1).mean())
v_rollinge = np.log(Evv_df.ewm(span=roll_window, axis=1).mean())
w_rollinge = np.log(Eww_df.ewm(span=roll_window, axis=1).mean())

# calculate gradient and intersect and separate into columns
col_m = ["Euu_m_e750", "Evv_m_e750", "Eww_m_e750"]
col_c = ["Euu_c_e750", "Evv_c_e750", "Eww_c_e750"]
for num, rolled in enumerate([u_rollinge, v_rollinge, w_rollinge]):
    mc = [
        np.polyfit(logk.loc[row], rolled.loc[row], 1) for row in list(readings_df.index)
    ]
    readings_df["temp"] = mc
    str1 = col_m[num]
    str2 = col_c[num]
    readings_df[[str1, str2]] = readings_df.temp.apply(pd.Series)

readings_df.drop(["temp"], axis=1, inplace=True)

# check
readings_df.shape
readings_df.head(1)

#############################################

######### get diff gradient #################
"""This creates exponentially smoothed data over window specified.
    Note window = 150 is 1 minute and 750 is 5 minutes"""
# choose smoothing paramaters
roll_window = 750
start = roll_window - 1
# create smoothing with chosen window
u_rollingd = np.log(Euu_df.rolling(window=roll_window, axis=1).mean())
v_rollingd = np.log(Evv_df.rolling(window=roll_window, axis=1).mean())
w_rollingd = np.log(Eww_df.rolling(window=roll_window, axis=1).mean())

# calculate gradient and intersect and separate into columns
col_m = ["Euu_m_d750", "Evv_m_d750", "Eww_m_d750"]
col_c = ["Euu_c_d750", "Evv_c_d750", "Eww_c_d750"]
for num, rolled in enumerate([u_rollingd, v_rollingd, w_rollingd]):
    mc = [
        np.polyfit(logk.loc[row][start:], rolled.loc[row][start:], 1)
        for row in list(readings_df.index)
    ]
    readings_df["temp"] = mc
    str1 = col_m[num]
    str2 = col_c[num]
    readings_df[[str1, str2]] = readings_df.temp.apply(pd.Series)
readings_df.drop(["temp"], axis=1, inplace=True)
readings_df.head(1)

# check
readings_df.shape
readings_df.head(1)

#############################################

######### gradients for decades #############
"""This creates gradients for 3 durations based on decades"""
# create decades
k_df["start_dec_lim"] = k_df["k0"].apply(lambda x: start_dec_lim(x))
k_df["end_dec_lim"] = k_df["k2210"].apply(lambda x: 10 ** (len(str(int(x)))))
k_df["no_decades"] = np.log10(k_df["end_dec_lim"]) - np.log10(k_df["start_dec_lim"])
k_df["end_dec_lim"] = k_df["k2210"].apply(lambda x: 10 ** (len(str(int(x)))))

#############################################

readings_df["u_mean"] = get_umean(u_df)

#############################################

