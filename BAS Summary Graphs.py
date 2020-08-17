# libraries
from mpl_toolkits.mplot3d import Axes3D

# create dataframe
draw = features2[cols].join(update2[["aggclu", "pairwise_distancespca"]])
draw.head(1)

# 3d plot
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
ax.scatter(
    draw["umean"],
    draw["Eww_m_e750"],
    draw["k_std_70new"],
    c=draw["aggclu"],
    cmap="Paired",
)
ax.set_xlabel("Mean of U TD")
ax.set_ylabel("Eww exp gradient")
plt.ylim((0.0, -2.0))
ax.set_zlabel("k std after 70")
ax.set_title("Clusters")

legend = ax.legend(*g.legend_elements(), loc="lower center", borderaxespad=-10, ncol=4)
ax.add_artist(legend)

# 3d plot
fig = plt.figure(figsize=(6, 6))
ax = Axes3D(fig)
scatter = ax.scatter(
    draw["k_std_70new"],
    draw["umean"],
    draw["Eww_m_e750"],
    c=draw["aggclu"],
    cmap="Accent",
)
legend = ax.legend(*scatter.legend_elements(), loc="upper left")
ax.set_title("Clusters")
ax.set_xlabel("k std after 70")
ax.set_ylabel("Mean of U TD")
ax.set_zlabel("Eww exp gradient")
ax.add_artist(legend)
legend_labels = ["A", "B", "C", "D", "E", "F", "G"]
for i in range(7):
    legend.get_texts()[i].set_text(legend_labels[i])
ax.add_artist(legend)
plt.show()


fig = plt.figure(figsize=(6, 6))
fig, ax = plt.subplots()
fig.set_figheight(10)
fig.set_figwidth(10)
scatter = plt.scatter(draw["umean"], draw["Eww_m_e750"], c=draw["gmm"], cmap="Accent")
ax.set_xlabel("Mean of U Time Domain (ms-1)")
ax.set_ylabel("Eww_k Exponential Smoothed Gradient")
ax.set_title("GMM Clusters")
# legend = ax.legend(*scatter.legend_elements())
# legend_labels = ["A", "B", "C", "D", "E", "F", "G"]
# for i in range(7):
#    legend.get_texts()[i].set_text(legend_labels[i])
# ax.add_artist(legend)
plt.show()

# violinplots
sns.violinplot(features2["Eww_m_e750"])

# scatter graphs with legend outside of plot
sns.scatterplot(
    readings_df["u_mean"],
    readings_df["Eww_m_e750"],
    hue=readings_df["gmm_20_tpca"],
    palette="muted",
)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

