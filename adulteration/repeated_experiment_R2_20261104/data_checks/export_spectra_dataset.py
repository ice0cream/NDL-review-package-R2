# -*- coding: utf-8 -*-
"""Write the NIR spectra of the cleaned file set as one table (one row per reading).

Input  : the cleaned OPUS files (built by fix_labels_by_time.py), read with the same reader as the analyses.
Output : ../spectra_dataset/NIR_spectra_by_reading.csv
         columns file, label, rep, time, then wn_<wavenumber> (apparent absorbance, log10(1/R), as exported by the
         instrument software). Values are written with 17 significant digits, so they read back bit-identical with float_precision='round_trip'.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'supporting_analysis'))
import redo_analysis as ra  # noqa: E402

OUT = HERE.parent / 'spectra_dataset' / 'NIR_spectra_by_reading.csv'


def main():
    files, S, axis = ra.load_spectra()
    cols = ['wn_' + repr(float(w)) for w in axis]
    d = pd.concat([files[['file', 'label', 'rep']].reset_index(drop=True),
                   pd.Series(files['time'].dt.strftime('%Y-%m-%d %H:%M:%S').to_numpy(), name='time'),
                   pd.DataFrame(S, columns=cols)], axis=1)
    OUT.parent.mkdir(exist_ok=True)
    d.to_csv(OUT, index=False, float_format='%.17g')
    back = pd.read_csv(OUT, float_precision='round_trip')
    assert np.array_equal(back[cols].to_numpy(float), S)
    assert np.array_equal(np.array([float(c[3:]) for c in cols]), axis)
    print('readings', len(d), '| wavenumbers', len(cols), '| written', OUT)


if __name__ == '__main__':
    main()
