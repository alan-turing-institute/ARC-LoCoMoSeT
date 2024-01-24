import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kendalltau

file = "results_bin_watch.csv"

df = pd.read_csv(file)

plt.style.use("default")
plt.rcParams["font.family"] = "Palatino"
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
plt.rc("grid", linestyle="-", color="black")
plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
s = 60
fig, ax = plt.subplots(2, 3, figsize=(16, 8))

ax[0][0].scatter(df["n_pars"], df["test/accuracy"], c="b", s=s)
ax[0][0].set_xlabel("Metric Value")
ax[0][0].set_ylabel("Test Accuracy")
tau_pars = kendalltau(df["n_pars"], df["test/accuracy"])[0]
ax[0][0].set_title(rf"No. Parameters ($\tau$ = {tau_pars:.2f})")

ax[0][1].scatter(df["renggli"], df["test/accuracy"], c="b", s=s)
ax[0][1].set_xlabel("Metric Value")
tau_renggli = kendalltau(df["renggli"], df["test/accuracy"])[0]
ax[0][1].set_title(rf"Renggli ($\tau$ = {tau_renggli:.2f})")

ax[0][2].scatter(df["LogME"], df["test/accuracy"], c="b", s=s)
ax[0][2].set_xlabel("Metric Value")
tau_logme = kendalltau(df["LogME"], df["test/accuracy"])[0]
ax[0][2].set_title(rf"LogME ($\tau$ = {tau_logme:.2f})")

ax[1][0].scatter(df["n_pars"].rank(), df["test/accuracy"].rank(), c="b", s=s)
ax[1][0].set_xlabel("Metric Rank")
ax[1][0].set_ylabel("Test Accuracy Rank")

ax[1][1].scatter(df["renggli"].rank(), df["test/accuracy"].rank(), c="b", s=s)
ax[1][1].set_xlabel("Metric Rank")

ax[1][2].scatter(df["LogME"].rank(), df["test/accuracy"].rank(), c="b", s=s)
ax[1][2].set_xlabel("Metric Rank")

fig.tight_layout()

fig.savefig("bin_watch.png", dpi=300)
