import string
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

ALPHA = 0.05
DPI = 600
POINT_SIZE = 38
JITTER_WIDTH = 0.08
BOX_WIDTH = 0.55
BOX_ALPHA = 0.45
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

    cleaned = sorted(cleaned, key=lambda x: (-len(x), [groups.index(v) for v in x]))
    letters = generate_letter_strings(len(cleaned))
    group_letters = {g: "" for g in groups}

    for letter, clique in zip(letters, cleaned):
        for g in clique:
            group_letters[g] += letter

    return group_letters


def analyze_one_variable(data, value_col, group_col, groups):
    summary_rows = []
    group_values = []

    for g in groups:
        vals = data.loc[data[group_col] == g, value_col].dropna().values
        group_values.append(vals)

        q1 = np.percentile(vals, 25) if len(vals) > 0 else np.nan
        q3 = np.percentile(vals, 75) if len(vals) > 0 else np.nan

        summary_rows.append({
            "Variable": value_col,
            "Group": g,
            "n": len(vals),
            "Mean": np.mean(vals) if len(vals) > 0 else np.nan,
            "SD": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "Median": np.median(vals) if len(vals) > 0 else np.nan,
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1 if len(vals) > 0 else np.nan,
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

    tmp = data.loc[data[value_col].notna(), [group_col, value_col]].copy()

    if tmp[group_col].nunique() >= 2:
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
                flag = False if pd.isna(pval) else pval >= ALPHA

                if g1 in nonsig.index and g2 in nonsig.columns:
                    nonsig.loc[g1, g2] = flag
                    nonsig.loc[g2, g1] = flag

            group_letters = compact_letter_display(groups, nonsig)
            letters_df["Letter"] = letters_df["Group"].map(group_letters)

        except Exception as e:
            warnings.warn(f"Tukey analysis failed for {value_col}: {e}")
            tukey_rows = pd.DataFrame(
                columns=["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject", "Variable"]
            )
    else:
        tukey_rows = pd.DataFrame(
            columns=["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject", "Variable"]
        )

    return summary_df, anova_row, pd.DataFrame(tukey_rows), letters_df


def plot_box_scatter_with_letters(ax, data, value_col, letters_df, group_col, groups, color_map):
    x = np.arange(len(groups))
    values_by_group = [
        data.loc[data[group_col] == g, value_col].dropna().values
        for g in groups
    ]

    bp = ax.boxplot(
        values_by_group,
        positions=x,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.4),
        boxprops=dict(linewidth=1.0, color="black"),
        whiskerprops=dict(linewidth=1.0, color="black"),
        capprops=dict(linewidth=1.0, color="black")
    )

    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(color_map[g])
        patch.set_alpha(BOX_ALPHA)

    all_vals = []
    max_values = []

    for i, g in enumerate(groups):
        vals = values_by_group[i]
        color = color_map[g]
        if len(vals) == 0:
            max_values.append(np.nan)
            continue

        all_vals.extend(vals)
        max_values.append(np.max(vals))

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

    if len(all_vals) == 0:
        return

    y_min = np.nanmin(all_vals)
    y_max = np.nanmax(all_vals)
    y_range = y_max - y_min if y_max > y_min else 1
    text_offset = y_range * 0.06

    for i, g in enumerate(groups):
        letter_vals = letters_df.loc[letters_df["Group"] == g, "Letter"].values
        letter = letter_vals[0] if len(letter_vals) > 0 else ""
        y_text = max_values[i] + text_offset if pd.notna(max_values[i]) else y_max + text_offset
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

    current_ylim = ax.get_ylim()
    ax.set_ylim(current_ylim[0], max(current_ylim[1], y_max + y_range * 0.18))


def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "03_general_parameters"
    result_dir = base_dir / "results" / "03_general_parameters" / "box_scatter"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "general_parameters.csv"
    df = pd.read_csv(file_path)

    group_col = None
    for col in df.columns:
        if str(col).lower() in ["group", "groups", "class", "label", "type"]:
            group_col = col
            break

    if group_col is None:
        raise ValueError("No grouping column found; name it group / class / label / type.")

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
        df["h°"] = h

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
        raise ValueError("No numeric columns found")

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
        summary_df, anova_df, tukey_df, letters_df = analyze_one_variable(df, value_col, group_col, groups)

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
            plot_box_scatter_with_letters(ax, df, value_col, letters_df, group_col, groups, color_map)

            fig.text(
                0.5, 0.01,
                "Different letters indicate significant differences (Tukey HSD, p < 0.05).",
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

            fig.savefig(result_dir / f"{safe_name}_box_scatter_letters.png", dpi=DPI, bbox_inches="tight")
            fig.savefig(result_dir / f"{safe_name}_box_scatter_letters.pdf", dpi=DPI, bbox_inches="tight")
            fig.savefig(result_dir / f"{safe_name}_box_scatter_letters.tif", dpi=DPI, bbox_inches="tight")
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
            plot_box_scatter_with_letters(
                ax,
                df,
                value_col,
                analysis_cache[value_col]["letters"],
                group_col,
                groups,
                color_map
            )

        for ax in axes[len(numeric_cols):]:
            ax.axis("off")

        fig.text(
            0.5, 0.005,
            "Different letters indicate significant differences among groups (Tukey HSD, p < 0.05).",
            ha="center", va="bottom", fontsize=9
        )

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        fig.savefig(result_dir / "All_variables_box_scatter_letters.png", dpi=DPI, bbox_inches="tight")
        fig.savefig(result_dir / "All_variables_box_scatter_letters.pdf", dpi=DPI, bbox_inches="tight")
        fig.savefig(result_dir / "All_variables_box_scatter_letters.tif", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    summary_all_df = pd.concat(summary_all, ignore_index=True)
    anova_all_df = pd.concat(anova_all, ignore_index=True)
    tukey_all_df = pd.concat(tukey_all, ignore_index=True)
    letters_all_df = pd.concat(letters_all, ignore_index=True)

    final_summary = letters_all_df.copy()
    final_summary["Mean ± SD"] = final_summary.apply(
        lambda r: f"{r['Mean']:.3f} ± {r['SD']:.3f}" if pd.notna(r["Mean"]) and pd.notna(r["SD"]) else "",
        axis=1
    )
    final_summary["Median (Q1–Q3)"] = final_summary.apply(
        lambda r: f"{r['Median']:.3f} ({r['Q1']:.3f}–{r['Q3']:.3f})" if pd.notna(r["Median"]) and pd.notna(r["Q1"]) and pd.notna(r["Q3"]) else "",
        axis=1
    )

    summary_all_df.to_csv(result_dir / "summary_statistics.csv", index=False, encoding="utf-8-sig")
    anova_all_df.to_csv(result_dir / "anova_results.csv", index=False, encoding="utf-8-sig")
    tukey_all_df.to_csv(result_dir / "tukey_pairwise_results.csv", index=False, encoding="utf-8-sig")
    letters_all_df.to_csv(result_dir / "group_letters.csv", index=False, encoding="utf-8-sig")
    final_summary.to_csv(result_dir / "final_summary_with_letters.csv", index=False, encoding="utf-8-sig")

    excel_path = result_dir / "box_scatter_statistics_results.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        summary_all_df.to_excel(writer, sheet_name="Summary_statistics", index=False)
        anova_all_df.to_excel(writer, sheet_name="ANOVA", index=False)
        tukey_all_df.to_excel(writer, sheet_name="Tukey_pairwise", index=False)
        letters_all_df.to_excel(writer, sheet_name="Letters", index=False)
        final_summary.to_excel(writer, sheet_name="Final_summary", index=False)

    print("Done")
    print(f"Output folder: {result_dir}")
    print("Generated:")
    print("1. Individual box-scatter plots with letters")
    print("2. Combined panel plot All_variables_box_scatter_letters")
    print("3. summary_statistics.csv")
    print("4. anova_results.csv")
    print("5. tukey_pairwise_results.csv")
    print("6. group_letters.csv")
    print("7. final_summary_with_letters.csv")
    print("8. box_scatter_statistics_results.xlsx")
    print("Variables:")
    print(numeric_cols)


if __name__ == "__main__":
    main()
