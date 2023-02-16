import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plot showing the distribution of ICD-9 and ICD-10 codes in MIMIC-IV


mimiciv_hosp = pd.read_csv(
    "path/to/physionet.org/files/mimiciv/2.2/hosp/diagnoses_icd.csv.gz",
    compression="gzip",
)

sns.set_theme("paper", style="whitegrid", palette="colorblind", font_scale=1.5)

icd9_counts = mimiciv_hosp[mimiciv_hosp.icd_version == 9].icd_code.value_counts()
icd10_counts = mimiciv_hosp[mimiciv_hosp.icd_version == 10].icd_code.value_counts()

fig = plt.figure(figsize=(10, 5))
ax = sns.lineplot(
    x=np.linspace(0, len(icd9_counts) - 1, len(icd9_counts)),
    y=icd9_counts.values,
    label="ICD-9",
    linewidth=1.5,
)
sns.lineplot(
    x=np.linspace(0, len(icd10_counts) - 1, len(icd10_counts)),
    y=icd10_counts.values,
    label="ICD-10",
    ax=ax,
    linewidth=1.5,
)
ax.set(
    yscale="log",
    xlabel="Code index",
    ylabel="Frequency of code in MIMIC-IV",
)
ax.hlines(10, 0, 16200, linestyles="dashed", label="100", color="gray")
plt.xlim(-50, 16200)
fig.savefig(
    "files/images/mimiciv_code_freq.png",
    dpi=600,
    bbox_inches="tight",
)
