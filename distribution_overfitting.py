import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

plt.rcParams['figure.figsize'] = [12, 8]
c = ["#007fff", "#ff3616", "#138d75", "#7d3c98", "#fbea6a"]  # Blue, Red, Green, Purple, Yellow
cset = ["#fdb462", "#fb8072"]
set2 = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]  # Set2
div = [
    "#a6cee3",
    "#1f78b4",
    "#b2df8a",
    "#33a02c",
]
lines = [set2[0], set2[4]]
lines = ["#e9a3c9", "#a1d76a"]
lines = ["#0A0A0A", "#0A0A0A"]
linestyles = ["solid", "dashed"]

sns.set(
    style="ticks",
    rc={
        "font.family": "Arial",
        "font.size": 40,
        "axes.linewidth": 2,
        "lines.linewidth": 3,
    },
    font_scale=2.5,
    palette=sns.color_palette("Set2")
)

np.random.seed(2021)
samples = 15
I_test = 1000
noise = 3

b, a = -10, 10
total_RMSEP = []
total_RMSEC = []
for _ in tqdm(np.arange(1024)):
    x = (b - a) * np.random.rand(samples) + a
    y = x**3 + ((b - a) * np.random.randn(samples) + a)*noise
    x_axis = np.linspace(np.min(x), np.max(x), 200)

    x_test = (b - a) * np.random.rand(I_test) + a
    y_test = x_test**3 + ((b - a) * np.random.randn(I_test) + a)*noise
    x_test_axis = np.linspace(np.min(x_test), np.max(x_test), 200)

    RMSEC, RMSEP = [], []
    max_order = 8
    for idx, order in enumerate(np.arange(1, max_order + 1)):
        p = np.polyfit(x, y, order)
        y_func = np.poly1d(p)

        RMSEP.append(np.sqrt(np.mean((y_test - y_func(x_test))**2)))
        RMSEC.append(np.sqrt(np.mean((y - y_func(x))**2)))

    total_RMSEP.append(RMSEP)
    total_RMSEC.append(RMSEC)
total_RMSEP = np.array(total_RMSEP)
total_RMSEC = np.array(total_RMSEC)
# fig, ax = plt.subplots()
RMSEP_means = total_RMSEP.sum(axis=0)/len(total_RMSEP)
RMSEC_means = total_RMSEC.sum(axis=0)/len(total_RMSEC)
colors = sns.color_palette("Set2", max_order)

px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(
    1, 2, sharex=True, figsize=(1848*px, 965*px)
)
sns.boxplot(data=total_RMSEC, ax=ax[0])
sns.boxplot(data=total_RMSEP, ax=ax[1])
ax[0].set_xticks(np.arange(0, max_order))
ax[0].set_xticklabels(np.arange(1, max_order + 1))
ax[0].set_yscale("log")
# ax[0].set_ylim(5e0, 1e5)
# ax[1].set_ylim(1e-2, 1e5)

# inset image
axins = ax[1].inset_axes([0.2, 0.2, 0.57, 0.67])
sns.boxplot(data=total_RMSEP, ax=axins)
x1, x2, y1, y2 = -.5, 7.5, 5e1, 6e2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
# axins.set_yticklabels([])

ax[1].indicate_inset_zoom(axins, edgecolor="k")

ax[0].set_ylabel("Error")
fig.text(0.5, 0.04, "Polynomial order", ha="center", va="center")

ax[0].set_title("Training error")
ax[1].set_title("Test error")
fig.subplots_adjust(wspace=0.1)
plt.savefig("./distribution_overfitting.pdf")
plt.show()
