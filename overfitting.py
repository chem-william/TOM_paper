import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


plt.rcParams['figure.figsize'] = [12, 8]
c = [
    "#007fff",
    "#ff3616",
    "#138d75",
    "#7d3c98",
    "#fbea6a"
]  # Blue, Red, Green, Purple, Yellow

lines = sns.color_palette("Accent", 12)
lines = [lines[2], lines[0], lines[4]]
linestyles = ["solid"]*3

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
set2 = sns.color_palette("Set2", 5)

np.random.seed(2021)

samples = 15
I_test = 1000
noise = 3

total_errors = []
a, b = -10, 10
x = (b - a) * np.random.rand(samples) + a
y = x**3 + ((b - a) * np.random.randn(samples) + a)*noise
x_axis = np.linspace(np.min(x), np.max(x), 200)

a, b = -10, 10
x_test = (b - a) * np.random.rand(I_test) + a
y_test = x_test**3 + ((b - a) * np.random.randn(I_test) + a)*noise
x_test_axis = np.linspace(np.min(x_test), np.max(x_test), 200)

RMSEC, RMSEP = [], []
max_order = 8
px = 1/plt.rcParams['figure.dpi']
fig, axes = plt.subplots(1, 2, figsize=(1848*px, 965*px))
color_idx = 0
color_count = 0
for idx, order in enumerate(np.arange(1, max_order + 1)):
    p = np.polyfit(x, y, order)
    y_func = np.poly1d(p)

    RMSEP.append(np.sqrt(np.mean((y_test - y_func(x_test))**2)))
    RMSEC.append(np.sqrt(np.mean((y - y_func(x))**2)))

    if order in (1, 3, max_order):
        axes[0].plot(
            x_test_axis,
            y_func(x_test_axis),
            label=f"Polynomial order: {order}",
            c=set2[color_count],
            # c=lines[color_idx],
            linestyle=linestyles[color_idx],
            linewidth=30,
            alpha=0.5,
            zorder=max_order - order,
        )
        color_idx += 1
        color_count += 1

    if idx == 0:
        axes[0].scatter(
            x,
            y,
            label="Training set",
            marker="X",
            linewidth=.7,
            c=set2[color_count],
            edgecolor="w",
            s=240,
            zorder=1001,
        )
        # axes[0].scatter(
        #     x,
        #     y,
        #     label="Training set",
        #     marker="x",
        #     c="white",
        #     s=256,
        #     zorder=99,
        # )
        color_count += 1
        axes[0].scatter(
            x_test,
            y_test,
            label="Test set",
            # c=c[0],
            c=set2[color_count],
            s=20,
            # alpha=0.8,
            edgecolor="w",
            linewidth=.4,
            zorder=1000,
        )
        color_count += 1

axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")

axbig = axes[1]

x = np.arange(1, len(RMSEC) + 1)
axbig.bar(
    x - .13, RMSEP, width=0.25, label="Test", facecolor=set2[2], edgecolor="none"
)
axbig.bar(
    x + .13,
    RMSEC,
    width=0.25,
    label="Training",
    facecolor=set2[1],
    edgecolor="none"
)
axbig.axhline(np.min(RMSEP), linestyle="--", c="grey", label="Lowest test")
axbig.set_xlabel("Polynomial order")
axbig.set_ylabel("Error")
axbig.legend(frameon=False, loc="best")
axbig.set_xlim(.7, max_order + .3)
axbig.set_ylim(-.05, np.max(RMSEP) + np.max(RMSEP)*0.05)
labels = np.arange(1, max_order + 1)
axbig.set_xticks(labels)
axbig.set_xticklabels(labels)

handles, labels = axes[0].get_legend_handles_labels()
labels = ["1st order", "3rd order", "8th order", "Training set", "Test set"]
axes[0].legend(handles, labels, loc="lower right")

plt.subplots_adjust(wspace=0.2, hspace=0.0, top=0.95, left=0.06, right=0.99)

fig.savefig("overfitting.pdf")
plt.show()
