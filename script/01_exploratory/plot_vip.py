import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from common.plots import set_publication_style


USE_CVSE_BAND = False
FIG_WIDTH = 3.2
FIG_HEIGHT = 4.4
DPI = 600
COLOR_LOW = "#A8D5BA"
COLOR_HIGH = "#4F81BD"
COLOR_THRESHOLD = "#D62728"
COLOR_BAND = "#BFD7EA"


def extract_wavenumber(x):
    x = str(x).replace("wn_", "")
    nums = re.findall(r"\d+\.?\d*", x)
    if len(nums) == 0:
        return np.nan
    return float(nums[0])


def main():
    set_publication_style()
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "vip_table.xlsx"
    if not file_path.exists():
        file_path = data_dir / "vip_table.csv"

    if file_path.suffix.lower() == ".csv":
        df = pd.read_csv(file_path)
    else:
        xls = pd.ExcelFile(file_path)
        if "General List-VIP" in xls.sheet_names:
            df = pd.read_excel(file_path, sheet_name="General List-VIP")
        else:
            df = pd.read_excel(file_path, sheet_name=xls.sheet_names[0])

    cols = df.columns.tolist()

    var_col = None
    for c in cols:
        if "Var ID" in str(c) or "Wavenumber" in str(c):
            var_col = c
            break
    if var_col is None:
        raise ValueError("Wavenumber column not found")

    vip_col = None
    for c in cols:
        c_str = str(c)
        if "VIP" in c_str and "cvSE" not in c_str and "*" not in c_str:
            vip_col = c
            break
    if vip_col is None:
        raise ValueError("VIP column not found")

    cvse_col = None
    for c in cols:
        c_str = str(c)
        if "cvSE" in c_str or "* M" in c_str:
            cvse_col = c
            break

    df["Wavenumber"] = df[var_col].apply(extract_wavenumber)
    df["VIP"] = pd.to_numeric(df[vip_col], errors="coerce")

    if cvse_col is not None:
        df["VIP_cvSE_scaled"] = pd.to_numeric(df[cvse_col], errors="coerce")
    else:
        df["VIP_cvSE_scaled"] = np.nan

    df = df.dropna(subset=["Wavenumber", "VIP"]).copy()
    df = df.sort_values("Wavenumber", ascending=True).reset_index(drop=True)

    wn = df["Wavenumber"].values
    vip = df["VIP"].values
    cvse = df["VIP_cvSE_scaled"].values

    out_table = result_dir / "vip_cleaned.csv"
    df[["Wavenumber", "VIP", "VIP_cvSE_scaled"]].to_csv(out_table, index=False, encoding="utf-8-sig")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Times"]
    plt.rcParams["axes.linewidth"] = 1.0

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=DPI)

    if USE_CVSE_BAND and not np.all(np.isnan(cvse)):
        vip_left = np.clip(vip - cvse, 0, None)
        vip_right = vip + cvse
        ax.fill_betweenx(wn, vip_left, vip_right, color=COLOR_BAND, alpha=0.25, linewidth=0)

    vip_low = np.where(vip <= 1.0, vip, np.nan)
    vip_high = np.where(vip > 1.0, vip, np.nan)

    ax.plot(vip_low, wn, color=COLOR_LOW, linewidth=1.4, solid_capstyle="round")
    ax.plot(vip_high, wn, color=COLOR_HIGH, linewidth=1.5, solid_capstyle="round")

    ax.axvline(x=1.0, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.0)

    y_top = wn.max()
    y_bottom = wn.min()
    y_range = y_top - y_bottom

    ax.text(1.08, y_top - 0.04 * y_range, "VIP>1.0", color=COLOR_THRESHOLD, fontsize=8.5, fontweight="bold", ha="left", va="top")
    ax.text(1.0, y_bottom + 0.02 * y_range, "1.0", color=COLOR_THRESHOLD, fontsize=7.5, ha="center", va="bottom",
            bbox=dict(facecolor="white", edgecolor=COLOR_THRESHOLD, boxstyle="square,pad=0.12", linewidth=0.8))

    ax.set_xlabel("VIP value", fontsize=10, fontweight="bold")
    ax.set_ylabel("Wavenumber/cm$^{-1}$", fontsize=10, fontweight="bold")

    xmax = max(2.0, np.nanmax(vip) + 0.1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(wn.min(), wn.max())

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    if (wn.max() - wn.min()) > 5000:
        ax.yaxis.set_major_locator(MultipleLocator(1000))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(500))

    ax.tick_params(axis="both", which="major", labelsize=8, width=1.0, length=3.5, direction="out")
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    suffix = "_journal_style"
    if USE_CVSE_BAND:
        suffix += "_with_cvse"

    out_png = result_dir / f"vip_plot{suffix}.png"
    out_pdf = result_dir / f"vip_plot{suffix}.pdf"
    out_tif = result_dir / f"vip_plot{suffix}.tif"

    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_tif, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    vip_gt1 = df[df["VIP"] > 1.0].copy()
    vip_gt1_out = result_dir / "vip_greater_than_1.csv"
    vip_gt1.to_csv(vip_gt1_out, index=False, encoding="utf-8-sig")

    print(f"VIP variables > 1.0: {len(vip_gt1)}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_tif}")


if __name__ == "__main__":
    main()
