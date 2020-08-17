# libraries
import glob
import pandas as pd
import numpy as np
from scipy.io import loadmat

pd.set_option("display.max_rows", 500)

############## load datasets ################

# Get all the file names
mat_files_name = glob.glob("data/Larsen*")
num_mat_files = len(mat_files_name)

# Load all the files
mat_files = [loadmat(x) for x in mat_files_name]

## create dataframes for wave variables
features = ["Euu_k", "Evv_k", "Eww_k", "k"]
col_names = ["Euu", "Evv", "Eww", "k"]
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

# check dataframes and save
Evv_df.head()
Evv_df.shape

Euu_df.to_csv("Euu_df.csv")
Evv_df.to_csv("Evv_df.csv")
Eww_df.to_csv("Eww_df.csv")
k_df.to_csv("k_df.csv")

## create dataframes for time variables
features = ["u", "v", "w"]
col_names = ["u", "v", "w"]
tdfs = []
for num, feat in enumerate(features):
    data = [mat_files[i][feat].reshape(-1).transpose() for i in range(num_mat_files)]
    combined = np.stack(data)
    cols = [str(col_names[num]) + str(i) for i in range(4426)]
    df = pd.DataFrame(combined, index=mat_files_name, columns=cols)
    df["item"] = df.index.str.split("\\").str[1]
    df = df.set_index("item")
    tdfs.append(df)

u_df = tdfs[0]
v_df = tdfs[1]
w_df = tdfs[2]

# checks and save
v_df.head()
w_df.shape

u_df.to_csv("u_df.csv")
v_df.to_csv("v_df.csv")
w_df.to_csv("w_df.csv")


# combine wave readings Euu, Evv and Eww
wave_df = Euu_df.join(Evv_df).join(Eww_df)
time_df = u_df.join(v_df).join(w_df)


# checks
wave_df.head()
wave_df.shape

time_df.head()
time_df.shape
