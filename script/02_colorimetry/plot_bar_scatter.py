import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import string
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from common.plots import set_publication_style


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = PROJECT_ROOT / "script" / "data" / "colorimetry.csv"
OUT_DIR = PROJECT_ROOT / "script" / "results" / "02_colorimetry" / "bar_scatter"

ALPHA = 0.05
DPI = 600
BAR_ALPHA = 0.70
POINT_SIZE = 38
JITTER_WIDTH = 0.08
SAVE_INDIVIDUAL_FIGS = True
SAVE_COMBINED_FIG = True
FIG_COLS = 3
USE_GROUP_ORDER_IN_FILE = True


def generate_letter_strings(n):
    letters = []
    alphabet = string.ascii_lowercase
    i = 0
    while len(letters) < n:
        x = i
        s = ""
        while True:
            s = alphabet[x % 26] + s
            x = x // 26 - 1
            if x < 0:
                break
        letters.append(s)
        i += 1
    return letters


def bron_kerbosch(R, P, X, adj, cliques):
    if not P and not X:
        cliques.append(sorted(R))
        return
    for v in list(P):
        bron_kerbosch(R | {v}, P & adj[v], X & adj[v], adj, cliques)
        P.remove(v)
        X.add(v)


def compact_letter_display(groups, nonsig_matrix):
    adj = {
        g: set(h for h in groups if h != g and nonsig_matrix.loc[g, h])
        for g in groups
    }

    cliques = []
    bron_kerbosch(set(), set(groups), set(), adj, cliques)

    for g in groups:
        if not any(g in c for c in cliques):
            cliques.append([g])

    cleaned = []
    seen = set()
    for c in cliques:
        c_tuple = tuple(sorted(c, key=lambda x: groups.index(x)))
        if c_tuple not in seen:
            cleaned.append(c_tuple)
            seen.add(c_tuple)

    cleaned = sorted(
        cleaned,
        key=lambda x: (-len(x), [groups.index(v) for v in x])
    )

    letters = generate_letter_strings(len(cleaned))
    group_letters = {g: "" for g in groups}

    for letter, clique in zip(letters, cleaned):
        for g in clique:
            group_letters[g] += letter

    return group_letters


def analyze_one_variable(df, groups, group_col, value_col):
    summary_rows = []
    group_values = []

    for g in groups:
        vals = df.loc[df[group_col] == g, value_col].dropna().values
        group_values.append(vals)

        summary_rows.append({
            "Variable": value_col,
            "Group": g,
            "n": len(vals),
            "Mean": np.mean(vals) if len(vals) > 0 else np.nan,
            "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "Min": np.min(vals) if len(vals) > 0 else np.nan,
            "Max": np.max(vals) if len(vals) > 0 else np.nan
        })

    summary_df = pd.DataFrame(summary_rows)

    valid_groups = [vals for vals in group_values if len(vals) >= 2]

    if len(valid_groups) >= 2:
        try:
            F_stat, p_anova = f_oneway(*valid_groups)
        except Exception:
            F_stat, p_anova = np.nan, np.nan
    else:
        F_stat, p_anova = np.nan, np.nan

    anova_row = pd.DataFrame([{
        "Variable": value_col,
        "Test": "One-way ANOVA",
        "F": F_stat,
        "P": p_anova
    }])

    tukey_rows = []
    letters_df = summary_df.copy()
    letters_df["Letter"] = ""

    valid_mask = df[value_col].notna()
    tmp = df.loc[valid_mask, [group_col, value_col]].copy()

    unique_valid_groups = tmp[group_col].unique().tolist()
    if len(unique_valid_groups) >= 2:
        try:
            tukey = pairwise_tukeyhsd(
                endog=tmp[value_col].values,
                groups=tmp[group_col].values,
                alpha=ALPHA
            )

            tukey_table = pd.DataFrame(
                tukey.summary().data[1:],
                columns=tukey.summary().data[0]
            )
            tukey_table["Variable"] = value_col

            tukey_table["meandiff"] = pd.to_numeric(tukey_table["meandiff"], errors="coerce")
            tukey_table["p-adj"] = pd.to_numeric(tukey_table["p-adj"], errors="coerce")

            tukey_rows = tukey_table.copy()

            nonsig = pd.DataFrame(True, index=groups, columns=groups)

            for _, r in tukey_table.iterrows():
                g1 = str(r["group1"])
                g2 = str(r["group2"])
                pval = r["p-adj"]

                if pd.isna(pval):
                    flag = False
                else:
                    flag = pval >= ALPHA

                if g1 in nonsig.index and g2 in nonsig.columns:
                    nonsig.loc[g1, g2] = flag
                    nonsig.loc[g2, g1] = flag

            group_letters = compact_letter_display(groups, nonsig)
            letters_df["Letter"] = letters_df["Group"].map(group_letters)

        except Exception as e:
            warnings.warn(f"Tukey analysis failed for {value_col}: {e}")
            tukey_rows = pd.DataFrame(columns=["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject", "Variable"])

    else:
        tukey_rows = pd.DataFrame(columns=["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject", "Variable"])

    return summary_df, anova_row, pd.DataFrame(tukey_rows), letters_df


def plot_bar_scatter_with_letters(ax, df, groups, group_col, value_col, letters_df, color_map):
    x = np.arange(len(groups))

    means = []
    sds = []
    max_values = []

    for g in groups:
        vals = df.loc[df[group_col] == g, value_col].dropna().values
        means.append(np.mean(vals) if len(vals) > 0 else np.nan)
        sds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0)
        max_values.append(np.max(vals) if len(vals) > 0 else np.nan)

    means = np.array(means, dtype=float)
    sds = np.array(sds, dtype=float)
    max_values = np.array(max_values, dtype=float)

    for i, g in enumerate(groups):
        vals = df.loc[df[group_col] == g, value_col].dropna().values
        color = color_map[g]

        ax.bar(
            i,
            means[i],
            yerr=sds[i],
            width=0.62,
            capsize=4,
            color=color,
            alpha=BAR_ALPHA,
            edgecolor="black",
            linewidth=0.8,
            error_kw=dict(lw=1.0, capthick=1.0)
        )

        rng = np.random.default_rng(2026 + i)
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=POINT_SIZE,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=3
        )

    all_vals = []
    for g in groups:
        vals = df.loc[df[group_col] == g, value_col].dropna().values
        if len(vals) > 0:
            all_vals.append(vals)

    if len(all_vals) > 0:
        concat_vals = np.concatenate(all_vals)
        y_min = np.nanmin(concat_vals)
        y_max = np.nanmax(concat_vals)
    else:
        y_min, y_max = 0, 1

    y_range = y_max - y_min if y_max > y_min else 1
    text_offset = y_range * 0.06

    for i, g in enumerate(groups):
        letter = letters_df.loc[letters_df["Group"] == g, "Letter"].values[0]
        y_text = max(means[i] + sds[i], max_values[i]) + text_offset
        ax.text(
            i,
            y_text,
            letter,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel(str(value_col), fontsize=11)
    ax.set_title(str(value_col), fontsize=12, pad=8)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    set_publication_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if INPUT_FILE.suffix.lower() == ".csv":
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)

    group_col = None
    for col in df.columns:
        if str(col).lower() in ["group", "groups", "class", "label", "type"]:
            group_col = col
            break

    if group_col is None:
        raise ValueError("No grouping column found. Name it group / class / label / type.")

    df[group_col] = df[group_col].astype(str)

    if USE_GROUP_ORDER_IN_FILE:
        groups = pd.unique(df[group_col]).tolist()
    else:
        groups = sorted(df[group_col].unique().tolist())

    cols_lower = {str(c).lower(): c for c in df.columns}
    if all(k in cols_lower for k in ["l", "a", "b"]):
        L_col = cols_lower["l"]
        a_col = cols_lower["a"]
        b_col = cols_lower["b"]

        df["C*"] = np.sqrt(df[a_col] ** 2 + df[b_col] ** 2)

        h = np.degrees(np.arctan2(df[b_col], df[a_col]))
        h = np.where(h < 0, h + 360, h)
        df["h"] = h

    exclude_cols = {group_col}
    for possible_id in ["sample_id", "sample", "id", "name"]:
        for col in df.columns:
            if str(col).lower() == possible_id:
                exclude_cols.add(col)

    numeric_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue

        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
            numeric_cols.append(col)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found for analysis.")

    default_colors = [
        "#4C78A8", "#59A14F", "#B279A2", "#F28E2B",
        "#E15759", "#76B7B2", "#EDC948", "#9C755F"
    ]
    color_map = {g: default_colors[i % len(default_colors)] for i, g in enumerate(groups)}

    summary_all = []
    anova_all = []
    tukey_all = []
    letters_all = []

    analysis_cache = {}

    for value_col in numeric_cols:
        summary_df, anova_df, tukey_df, letters_df = analyze_one_variable(df, groups, group_col, value_col)

        summary_all.append(summary_df)
        anova_all.append(anova_df)
        tukey_all.append(tukey_df)
        letters_all.append(letters_df)

        analysis_cache[value_col] = {
            "summary": summary_df,
            "anova": anova_df,
            "tukey": tukey_df,
            "letters": letters_df
        }

        if SAVE_INDIVIDUAL_FIGS:
            fig, ax = plt.subplots(figsize=(5.0, 4.2), dpi=DPI)
            plot_bar_scatter_with_letters(ax, df, groups, group_col, value_col, letters_df, color_map)

            fig.text(
                0.5, 0.01,
                "Bars with different letters indicate significant differences (Tukey HSD, p < 0.05).",
                ha="center", va="bottom", fontsize=8
            )

            plt.tight_layout(rect=[0, 0.04, 1, 1])

            safe_name = (
                str(value_col)
                .replace("/", "_")
                .replace("\\", "_")
                .replace("*", "star")
                .replace("°", "deg")
            )

            fig.savefig(OUT_DIR / f"{safe_name}_bar_scatter_letters.png", dpi=DPI, bbox_inches="tight")
            fig.savefig(OUT_DIR / f"{safe_name}_bar_scatter_letters.pdf", bbox_inches="tight")
            fig.savefig(OUT_DIR / f"{safe_name}_bar_scatter_letters.tif", dpi=DPI, bbox_inches="tight")
            plt.close(fig)

    if SAVE_COMBINED_FIG:
        n_vars = len(numeric_cols)
        n_cols = FIG_COLS if n_vars >= FIG_COLS else n_vars
        n_rows = int(np.ceil(n_vars / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.6 * n_cols, 3.9 * n_rows),
            dpi=DPI
        )

        if n_vars == 1:
            axes = np.array([axes])
        axes = np.array(axes).reshape(-1)

        for ax, value_col in zip(axes, numeric_cols):
            plot_bar_scatter_with_letters(
                ax,
                df,
                groups,
                group_col,
                value_col,
                analysis_cache[value_col]["letters"],
                color_map
            )

        for ax in axes[len(numeric_cols):]:
            ax.axis("off")

        fig.text(
            0.5, 0.005,
            "Bars with different letters indicate significant differences among groups (Tukey HSD, p < 0.05).",
            ha="center", va="bottom", fontsize=9
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        fig.savefig(OUT_DIR / "All_variables_bar_scatter_letters.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(OUT_DIR / "All_variables_bar_scatter_letters.pdf", bbox_inches="tight")
        fig.savefig(OUT_DIR / "All_variables_bar_scatter_letters.tif", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    summary_all_df = pd.concat(summary_all, ignore_index=True)
    anova_all_df = pd.concat(anova_all, ignore_index=True)
    tukey_all_df = pd.concat(tukey_all, ignore_index=True)
    letters_all_df = pd.concat(letters_all, ignore_index=True)

    final_summary = letters_all_df.copy()
    final_summary["Mean_SD"] = final_summary.apply(
        lambda r: f"{r['Mean']:.3f} ± {r['SD']:.3f}" if pd.notna(r["Mean"]) and pd.notna(r["SD"]) else "",
        axis=1
    )

    summary_all_df.to_csv(OUT_DIR / "summary_statistics.csv", index=False, encoding="utf-8-sig")
    anova_all_df.to_csv(OUT_DIR / "anova_results.csv", index=False, encoding="utf-8-sig")
    tukey_all_df.to_csv(OUT_DIR / "tukey_pairwise_results.csv", index=False, encoding="utf-8-sig")
    letters_all_df.to_csv(OUT_DIR / "group_letters.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(OUT_DIR / "final_summary_with_letters.csv", index=False, encoding="utf-8-sig")

    excel_path = OUT_DIR / "bar_scatter_statistics_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_all_df.to_excel(writer, sheet_name="Summary_statistics", index=False)
        anova_all_df.to_excel(writer, sheet_name="ANOVA", index=False)
        tukey_all_df.to_excel(writer, sheet_name="Tukey_pairwise", index=False)
        letters_all_df.to_excel(writer, sheet_name="Letters", index=False)
        final_summary.to_excel(writer, sheet_name="Final_summary", index=False)

    print("Bar-scatter analysis completed.")
    print(f"Output folder: {OUT_DIR}")
    print("Generated files:")
    print("1. Individual bar-scatter plots with letters")
    print("2. Combined figure All_variables_bar_scatter_letters")
    print("3. summary_statistics.csv")
    print("4. anova_results.csv")
    print("5. tukey_pairwise_results.csv")
    print("6. group_letters.csv")
    print("7. final_summary_with_letters.csv")
    print("8. bar_scatter_statistics_results.xlsx")
    print("Variables analyzed:")
    print(numeric_cols)


if __name__ == "__main__":
    main()
