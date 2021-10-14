from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['figure.figsize'] = [12, 8]
sns.set(
    style="ticks",
    rc={
        "font.family": "Liberation Sans",
        "font.size": 40,
        "axes.linewidth": 2,
        "lines.linewidth": 3,
    },
    font_scale=2.5,
    palette=sns.color_palette("Set2")
)
lower, upper = -7, 0.5


def create_histograms(traces):
    hists = []
    for trace in traces:
        counts, binedges = np.histogram(trace, bins=128, range=(lower, upper))
        hists.append(counts)
    hists = np.array(hists)
    return hists


def truncate_trace(trace, threshold=-8):
    condition = trace < threshold
    counts = np.cumsum(condition)
    idx = np.searchsorted(counts, 3)
    trace = trace[:idx]

    return trace


def load_traces(npy_file: str):
    raw_traces = []
    data = np.load(npy_file, allow_pickle=True)
    for trace in tqdm(data):
        if len(trace) > 10000:
            trace = trace[10000:]
        if len(trace) > 3000:
            continue
        trace = np.log10(trace)
        thresh = np.argmax(trace <= upper)
        trace = trace[thresh:]
        trace = truncate_trace(trace, threshold=lower)
        raw_traces.append(trace)

    raw_traces = np.array(raw_traces)

    hists = create_histograms(raw_traces)

    return hists, raw_traces


# Load molecule A samples
A_npy_file = "./data_joseph/pd58.npy"
A_hists, A_traces = load_traces(A_npy_file)

px = 1/plt.rcParams['figure.dpi']
fig, ax = plt.subplots(1, 3, sharey=True, figsize=(1848*px, 965*px))
np.random.seed(2021)
samples = 5
offset = 12
delta_z = 0.002  # nm
delta_z *= 10  # Å
rand_ints = np.random.randint(0, len(A_traces), size=samples)
xaxis = np.linspace(lower, upper, 128)
for idx, rint in enumerate(rand_ints):
    ax[0].plot(
        delta_z * np.arange(len(A_traces[rint])) + offset*idx,
        A_traces[rint],
    )
    ax[1].plot(A_hists[rint] + offset*idx*8, xaxis)
ax[0].set_xlabel(r"$\Delta$z (Å)")
ax[0].set_ylabel(r"Conductance (log$_{10}$(G/G$_0$))")
ax[0].set_xticklabels([])

scalebar = AnchoredSizeBar(
    ax[0].transData,
    5,
    "5 ang",
    "lower left",
    pad=0.2,
    color="black",
    size_vertical=.05,
)
ax[0].add_artist(scalebar)

ax[1].set_xlabel("Counts")

ax[2].plot(A_hists.sum(axis=0), xaxis, c="k")
ax[2].set_xlabel("Counts")
ax[2].ticklabel_format(axis="x", scilimits=(0, 5))
ax[2].set_xlim(0, 160_000)

ax[0].set_ylim(lower, upper)

plt.subplots_adjust(wspace=0.05, left=0.06, right=0.99, top=0.99)
plt.legend(frameon=False)
plt.savefig("clustering_example_traces.pdf")
plt.show()
