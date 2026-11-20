import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from common.preprocessing import build_preprocessing_manifest, parse_wavenumber_columns


OUTPUT_DESCENDING = True


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    result_dir = Path(__file__).resolve().parent.parent / "results" / "01_exploratory"
    result_dir.mkdir(parents=True, exist_ok=True)

    file_path = data_dir / "nir_spectra.csv"
    df = pd.read_csv(file_path)

    wn_cols, wavenumbers = parse_wavenumber_columns(df, prefix="wn_")
    meta_df = df.drop(columns=wn_cols).reset_index(drop=True)

    sort_idx_asc = np.argsort(wavenumbers)
    wn_asc = wavenumbers[sort_idx_asc]
    X_raw = df[wn_cols].astype(float).values
    X_asc = X_raw[:, sort_idx_asc]

    out_dir = result_dir / "preprocessed_24_combinations"
    out_dir.mkdir(parents=True, exist_ok=True)

    combinations = build_preprocessing_manifest()
    manifest_records = []

    for combo in combinations:
        number = combo["Number"]
        path_method = combo["Pathlength"]
        spectra_method = combo["Spectra"]
        smoothing_method = combo["Smoothing"]

        from common.preprocessing import apply_pathlength, apply_spectral_processing
        X_step1 = apply_pathlength(X_asc, method=path_method)
        X_processed_asc = apply_spectral_processing(X_step1, wn_asc, spectra_method=spectra_method, smoothing_method=smoothing_method)

        if OUTPUT_DESCENDING:
            output_idx = np.argsort(wn_asc)[::-1]
        else:
            output_idx = np.argsort(wn_asc)

        wn_out = wn_asc[output_idx]
        X_out = X_processed_asc[:, output_idx]
        out_wn_cols = [f"wn_{w:.4f}" for w in wn_out]

        out_df = pd.concat([meta_df, pd.DataFrame(X_out, columns=out_wn_cols)], axis=1)
        method_name = f"{number:02d}_{path_method}_{spectra_method}_{smoothing_method}"
        out_csv = out_dir / f"nir_spectra_{method_name}.csv"
        out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

        manifest_records.append({
            "Number": number,
            "Pathlength": path_method,
            "Spectra": spectra_method,
            "Smoothing": smoothing_method,
            "File": out_csv.name
        })

    manifest_df = pd.DataFrame(manifest_records)
    manifest_csv = out_dir / "preprocessing_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    print(f"Preprocessed {len(combinations)} combinations")
    print(f"Output folder: {out_dir}")
    print(f"Manifest: {manifest_csv}")


if __name__ == "__main__":
    main()
