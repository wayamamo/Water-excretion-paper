import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_PATH = "/mnt/data/Pasted text(17).txt"

OUT_PNG = "/mnt/data/hydra_rp2_count30_leaky_overlay.png"
OUT_SVG = "/mnt/data/hydra_rp2_count30_leaky_overlay.svg"
OUT_PDF = "/mnt/data/hydra_rp2_count30_leaky_overlay.pdf"
OUT_CSV_LONG = "/mnt/data/hydra_rp2_count30_leaky_overlay_values_long.csv"
OUT_CSV_WIDE = "/mnt/data/hydra_rp2_count30_leaky_overlay_values_wide.csv"
OUT_EVENT_CSV = "/mnt/data/hydra_rp2_count30_leaky_overlay_events.csv"

TAU = 27.0
DT = 2.0
COUNT_WINDOW = 30.0

SHOW_EXCRETION_LINES = True
SHOW_AMBIGUOUS = True
AMBIGUOUS_EVENTS = [(5, 3649.0), (5, 9742.0)]

df = pd.read_csv(INPUT_PATH, sep="\t", header=None, engine="python")

groups = [
    (0, 1, 2),
    (5, 6, 7),
    (10, 11, 12),
    (15, 16, 17),
    (21, 22, 23),
]

records = []
ex_records = []

for gi, (a, b, c) in enumerate(groups, start=1):
    spikes = pd.to_numeric(df.iloc[2:, b], errors="coerce").dropna().values.astype(float)
    ex = pd.to_numeric(df.iloc[2:, c], errors="coerce").dropna().values.astype(float)
    for t in spikes:
        records.append({"imaging": gi, "spike_time": t})
    for t in ex:
        ex_records.append({"imaging": gi, "excretion_time": t, "ambiguous": False})

if SHOW_AMBIGUOUS:
    for gi, t in AMBIGUOUS_EVENTS:
        ex_records.append({"imaging": gi, "excretion_time": t, "ambiguous": True})

spikes_df = pd.DataFrame(records).sort_values(["imaging", "spike_time"]).reset_index(drop=True)
ex_df = pd.DataFrame(ex_records).sort_values(["imaging", "excretion_time"]).reset_index(drop=True)

def leaky_trace(spike_times, tgrid, tau):
    E = np.zeros_like(tgrid, dtype=float)
    for s in spike_times:
        idx = tgrid >= s
        E[idx] += np.exp(-(tgrid[idx] - s) / tau)
    return E

def rolling_count_trace(spike_times, tgrid, window):
    C = np.zeros_like(tgrid, dtype=float)
    for i, t in enumerate(tgrid):
        C[i] = np.sum((spike_times <= t) & (spike_times >= t - window))
    return C

all_rows = []
wide_tables = []
event_rows = []

for gi, g in spikes_df.groupby("imaging"):
    st = g["spike_time"].values
    ex_confirmed = ex_df[(ex_df.imaging == gi) & (~ex_df.ambiguous)]["excretion_time"].values
    ex_ambiguous = ex_df[(ex_df.imaging == gi) & (ex_df.ambiguous)]["excretion_time"].values

    end_time = max(
        st.max() if len(st) else 0,
        ex_confirmed.max() if len(ex_confirmed) else 0,
        ex_ambiguous.max() if len(ex_ambiguous) else 0
    )

    tgrid = np.arange(0, end_time + 1, DT)
    count30 = rolling_count_trace(st, tgrid, COUNT_WINDOW)
    leaky = leaky_trace(st, tgrid, TAU)

    sub = pd.DataFrame({
        "time_s": tgrid,
        f"img{gi}_count30": count30,
        f"img{gi}_leaky": leaky,
    })
    wide_tables.append(sub)

    for t, c, e in zip(tgrid, count30, leaky):
        all_rows.append({
            "imaging": gi,
            "time_s": t,
            "count30": c,
            "leaky_evidence": e,
        })

    for t in ex_confirmed:
        event_rows.append({"imaging": gi, "event_time_s": t, "event_type": "confirmed_excretion"})
    for t in ex_ambiguous:
        event_rows.append({"imaging": gi, "event_time_s": t, "event_type": "ambiguous_excretion"})

long_df = pd.DataFrame(all_rows)
long_df.to_csv(OUT_CSV_LONG, index=False)

wide_df = wide_tables[0]
for sub in wide_tables[1:]:
    wide_df = pd.merge(wide_df, sub, on="time_s", how="outer")
wide_df = wide_df.sort_values("time_s").reset_index(drop=True)
wide_df.to_csv(OUT_CSV_WIDE, index=False)

event_df = pd.DataFrame(event_rows).sort_values(["imaging", "event_time_s"])
event_df.to_csv(OUT_EVENT_CSV, index=False)

fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=False)

for ax, gi in zip(axes, sorted(spikes_df["imaging"].unique())):
    sub = long_df[long_df["imaging"] == gi]
    st = spikes_df[spikes_df.imaging == gi]["spike_time"].values
    ex_confirmed = ex_df[(ex_df.imaging == gi) & (~ex_df.ambiguous)]["excretion_time"].values
    ex_ambiguous = ex_df[(ex_df.imaging == gi) & (ex_df.ambiguous)]["excretion_time"].values

    ax.plot(sub["time_s"], sub["count30"], label="RP2 spike count in previous 30 s", linewidth=1.5)
    ax.plot(sub["time_s"], sub["leaky_evidence"], label=f"Leaky RP2 evidence (tau = {TAU:.0f} s)", linewidth=1.5)

    ymax = max(sub["count30"].max(), sub["leaky_evidence"].max(), 1.0)
    ax.vlines(st, 0, ymax * 0.06, linewidth=0.5, alpha=0.4)

    if SHOW_EXCRETION_LINES:
        for e in ex_confirmed:
            ax.axvline(e, linestyle="--", linewidth=1.0)
        for e in ex_ambiguous:
            ax.axvline(e, linestyle=":", linewidth=1.0)

    ax.set_ylabel(f"Img {gi}")

axes[0].legend(loc="upper right", fontsize=8, frameon=False)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("RP2 spike accumulation over time, 30 s spike count and leaky evidence")
fig.text(0.02, 0.5, "Accumulated RP2 activity", rotation=90, va="center")

plt.tight_layout(rect=[0.04, 0.02, 1, 0.96])

fig.savefig(OUT_PNG, dpi=300)
fig.savefig(OUT_SVG)
fig.savefig(OUT_PDF)
plt.close(fig)

print("Done.")
print("Saved:")
print(OUT_PNG)
print(OUT_SVG)
print(OUT_PDF)
print(OUT_CSV_LONG)
print(OUT_CSV_WIDE)
print(OUT_EVENT_CSV)
