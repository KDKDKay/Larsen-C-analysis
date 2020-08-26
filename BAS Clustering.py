# libraries
# region
#################################################
import numpy as np
import pandas as pd
import timeit
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.mixture import GaussianMixture
pd.set_option("display.max_rows", 500)
sns.set()
# endregion

############## load datasets ################
readings_df=timereadings_df
readings_df = pd.read_csv("time_df.csv", index_col='item')
Euu_df = pd.read_csv("raw_csvs\Euu_df.csv", index_col='item')
Evv_df = pd.read_csv("raw_csvs\Evv_df.csv", index_col='item')
Eww_df = pd.read_csv("raw_csvs\Eww_df.csv", index_col='item')
k_df = pd.read_csv("raw_csvs\k_df.csv", index_col='item')
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
    plt.title(f"{item} Spectra")
    plt.margins(0.25, 0.75)
    # gridlines
    gridline3y = np.exp((-5 / 3) * (np.log(k_df.loc[item])) - 9)
    gridline4y = np.exp((-5 / 3) * (np.log(k_df.loc[item])) - 13)
    gridline5y = np.exp((-5 / 3) * (np.log(k_df.loc[item])) - 17)
    plt.loglog(k_df.loc[item], gridline3y, c="gray", linestyle="dashed")
    plt.loglog(k_df.loc[item], gridline4y, c="gray", linestyle="dashed")
    plt.loglog(k_df.loc[item], gridline5y, c="gray", linestyle="dashed")
    plt.legend()

def draw_timegraph(item, u_df=u_df, v_df=v_df, w_df=w_df, time_df=time_df):
    """This code draws the log wave spectra graph
    with three gradient lines over a set range"""
    # draw graph
    f, ax = plt.subplots(figsize=(6, 6))
    plt.plot(time_df.loc[item], u_df.loc[item], label="u")
    plt.plot(time_df.loc[item], v_df.loc[item], label="v")
    plt.plot(time_df.loc[item], w_df.loc[item], label="w")
    # graph settings
    plt.title(f"{item} Time Domain")
    plt.margins(0.25, 0.75)
    plt.legend()


def mse(col, grad, intercept, idx):
    m = df[grad][idx]
    c = df[intercept][idx]
    logk = np.log(df["k"][idx])
    y_fit = np.exp(m * logk + c)
    return mean_squared_error(df[col][idx], y_fit)


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


def random_check(list_to_check, sample_size,  Euu_df=Euu_df, Evv_df=Evv_df, Eww_df=Eww_df, k_df=k_df):
    for item in random.sample(list_to_check, sample_size):
        draw_spectra(item)

#############################################

#################### Preprocessing ######################
# X = np.log(readings_df.values) use log for wave data
X = readings_df.values

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)
#scaled_X = np.log(scaled_X)

pca = PCA(n_components=0.8)
starttime = timeit.default_timer()
X_pca = pca.fit_transform(X)
print(timeit.default_timer() - starttime)
print(f"no pca dims: {X_pca.shape[1]}")

# check explained variance
pca.explained_variance_ratio_

# create pca df
pca_df = pd.DataFrame(
    X_pca,
    columns=["PC" + str(i + 1) for i in range(X_pca.shape[1])],
    index=readings_df.index,
)
pca_df.shape

pca_df.to_csv("pca_log.csv")

# check moat contributing feaatures
pca_feat_check = pd.DataFrame(
    pca.components_, columns=readings_df.columns, index=pca_df.columns
).T
pca_feat_check.head(1)
pca_feat_check.sort_values(by="PC1", ascending=True)
pca_feat_check['source'] = pca_feat_check.index.str[0]

# check the largest feature contributor by reading
source_check = pca_feat_check.iloc[pca_feat_check['PC1'].abs().argsort()]
source_check.reset_index(inplace=True)
source_check.head()
check = source_check.iloc[source_check['PC1'].abs().argsort()]
check[check['source']=='v'][-20:]


#################### Clustering ########################

#################### KMEANS ########################

numbers = range(3, 50)
kmeans_score = []
print("**********KMEANS********************")
for i in numbers:
    kmeans = KMeans(n_clusters=i)
    labels = kmeans.fit_predict(X_pca)
    print(np.bincount(labels))
    ss = silhouette_score(X_pca, labels)
    print(ss)
    kmeans_score.append(ss)
print("**********END********************")

plt.plot(numbers, kmeans_score)
plt.title("KMeans Silhouette Scores")

#############################################

#################### Pairwise ########################

all_distances = pairwise_distances(X_pca, metric="euclidean")
neig_distances = [np.min(row[np.nonzero(row)]) for row in all_distances]
neig_distances
# draw graph
plt.hist(neig_distances, bins=50)
plt.xlabel("Distance from closest sample")
plt.ylabel("Occurrences")
plt.title("Pairwise Distances for Time Domain PCA - reduced scale")
plt.ylim((0, 10))
readings_df['pairwise_pca_time'] = neig_distances
readings_df[readings_df['pairwise_pca_time']>0.1]


#############################################

############### Agglomorative Clustering #################

aggscore = []
print("**********DBSCAN********************")
for i in range(3, 50):
    n_cluster = i
    agglomerative = AgglomerativeClustering(
        n_clusters=n_cluster, linkage="ward", affinity="euclidean"
    )
    ss = silhouette_score(X_pca, cluster_assignment
    cluster_assignment = agglomerative.fit_predict(X_pca)
    aggscore.append())
    print(f"for {i}:")
    print(ss)
    print(np.bincount(cluster_assignment))
print("**********END********************")

plt.plot(range(3, 80), aggscore)
plt.title("Agglomorative Clustering Silhouette Scores")

#############################################

#################### DBSCAN ########################
"""EPS and sample size based on kmeans and pairwise analysis"""
noise_points = []
no_clusters = []
dbscan_score = []

eps = [0.3, 0.4, 0.5, 0.6]
min_samples = [25, 50, 75, 100]
arrtibute = ["auto", "ball_tree", "kd_tree", "brute"]

print("**********DBSCAN********************")
for ep in eps:
    for samp in min_samples:
        starttime = timeit.default_timer()
        clustering = DBSCAN(eps=ep, min_samples=samp).fit(X)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"for eps: {ep} and samples: {samp}:")
        print("Estimated number of clusters: %d" % n_clusters)
        print("Estimated number of noise points: %d" % n_noise)
        noise_points.append(n_noise)
        no_clusters.append(n_clusters)
        try:
            dbscan_score.append(silhouette_score(X, labels))
            print("Silhouette Coefficient: %0.3f" % silhouette_score(X, labels))
        except:
            dbscan_score.append("na")
            print("Silhouette Coefficient: na")
        print(f"Time to run: {timeit.default_timer()-starttime}")
print("**********END********************")

#############################################

#################### BIRCH ########################

birch_scores = []
threshold = [0.005, 0.0001, 0.0005, 0.00005]
branching_factor = [50, 100, 150]
n_clusters = [3, 5, 8, 11]

print("**********BIRCH CLUSTERING********************")
from sklearn.cluster import Birch

for branch in n_clusters:
    birch = Birch(threshold=0.0001, n_clusters=branch, branching_factor=50)
    cluster_assignment = birch.fit_predict(X_pca)
    try:
        birch_scores.append(silhouette_score(X_pca, cluster_assignment))
        print(f"for {branch}")
        print(np.bincount(cluster_assignment))
        print(f"Silhouette Coefficient: {silhouette_score(X_pca, cluster_assignment)}")
    except:
        birch_scores.append(f"{k}: na")
        print(f"for {branch}")
        print(f"Silhouette Coefficient: na")
print("**********END********************")
plt.title("Birch Score")
plt.plot(n_clusters, birch_scores)

#############################################

#################### GMM ########################

numbers = range(3, 50)
gmm_score = []
print("**********GMM********************")
for i in numbers:
    gmm = GaussianMixture(n_components=i)
    labels = gmm.fit_predict(X_pca)
    print(f"for {i}:")
    #print(np.bincount(labels))
    ss = silhouette_score(X_pca, labels)
    print(ss)
    gmm_score.append(ss)
print("**********END********************")

plt.plot(numbers, gmm_score)
plt.title("GMM Silhouette Scores Scaled")
plt.ylabel("Silhouette Score")
plt.xlabel("Number of bursts")

readings_df['gmm_20_tpca']=labels
#############################################

#################### checks ########################
# checks on the clusters to determine where there may be explanations

# random generation of graphs in a cluster to determine if they are degraded or not
random_check(list(readings_df[readings_df['gmm_20_tpca']==2].index), 5)

# drawing pairplots to determine correlations

#test = readings_df[readings_df['gmm_3_tpca']==2][['Eww_m_e750', 'Euu_m_e750', 'Evv_m_e750', 'u_mean', 'gmm_3_tpca']]
#test = readings_df[['Eww_m_e750', 'Euu_m_e750', 'Evv_m_e750', 'u_mean', 'gmm_20_tpca']]
test = readings_df[['Eww_m_e750', 'u_mean', 'gmm_20_tpca']]
test.shape

sns.pairplot(test, hue='gmm_20_tpca')
sns.scatterplot(readings_df['u_mean'], readings_df['Eww_m_e750'], hue=readings_df['gmm_20_tpca'], palette='muted')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("GMM Scaled PCA Time Domain")
plt.ylabel("Eww_k exponentially smoothed gradient")
plt.xlabel("Mean of U")

############################################################


readings_df.to_csv("time_domain_0817")

