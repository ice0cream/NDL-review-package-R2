from pathlib import Path
import re
import numpy as np
import pandas as pd
from brukeropusreader import read_file

try:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error
except ImportError as e:
    raise ImportError(
        "本脚本新增 PLSR + Hotelling T2 + Q residual 异常光谱剔除功能，"
        "需要安装 scikit-learn。可运行：pip install scikit-learn"
    ) from e

try:
    from scipy.stats import f as f_distribution
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False



# ============================================================
# 0. 使用说明
# ============================================================
# 目录建议：
# botanical_samples/
# ├─ DATA/                      # 放 .001 OPUS 文件
# ├─ 木薯淀粉.xlsx               # 放掺假比例表
# └─ 本脚本.py
#
# 文件名规则：
# cj_F批号;第几次遇见;技术重复_日期_时间.001
# 例如：cj_F9;2;3_20260523_115209.001
#
# Excel 表格规则：
# 编号 = F批号
# 第1次比例 -> 文件名里的 ;1;
# 第2次比例 -> 文件名里的 ;2;
# ...
# 第9次比例 -> 文件名里的 ;9;（脚本会自动识别所有“第n次比例”列）
#
# 光谱转换逻辑：
# Abs = -log10(ScSm / ScRf)
# 注意：不用 AB 块，适合 TANGO 文件 AB 块异常的情况。
# ============================================================


# ============================================================
# 1. 路径设置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# 自动寻找 .001 文件夹：
# 优先读取当前脚本同目录下的 DATA；
# 如果不存在，再尝试 木薯淀粉DATA；
# 如果仍不存在，则扫描当前目录下一层子文件夹。
INPUT_DIR_CANDIDATES = [
    BASE_DIR / "DATA",
    BASE_DIR / "木薯淀粉DATA",
    BASE_DIR / "玉米淀粉DATA",
]

INPUT_DIR = None
for _p in INPUT_DIR_CANDIDATES:
    if _p.exists() and len(list(_p.glob("*.001"))) > 0:
        INPUT_DIR = _p
        break

if INPUT_DIR is None:
    for _p in BASE_DIR.iterdir():
        if _p.is_dir() and len(list(_p.glob("*.001"))) > 0:
            INPUT_DIR = _p
            break

if INPUT_DIR is None:
    # 兜底：保持原路径名，后面主流程会报错并提示。
    INPUT_DIR = BASE_DIR / "DATA"

TABLE_PATH = BASE_DIR / "木薯淀粉.xlsx"

OUTPUT_DIR = BASE_DIR / "converted_csv_ratio_cassava"
MEAN_SPECTRA_DIR = OUTPUT_DIR / "mean_spectra_by_batch_encounter"
GROUP_SPECTRA_DIR = OUTPUT_DIR / "group_spectra_with_repeats"

OUTPUT_DIR.mkdir(exist_ok=True)
MEAN_SPECTRA_DIR.mkdir(exist_ok=True)
GROUP_SPECTRA_DIR.mkdir(exist_ok=True)

# 是否跳过 MU、QC、BK 等不符合 cj_Fx;x;x 格式的文件
SKIP_NON_CJ_F_FILES = True

# 是否把 cj_mu;1_日期_时间.001 这类纯木薯淀粉光谱作为 100% 掺假参考一起输出。
# 默认 False，避免纯粉样本拉宽浓度范围、影响 F 样品掺假比例模型。
INCLUDE_MU_REFERENCE = False

# 是否把 F/f 统一成大写 F
NORMALIZE_F_TO_UPPERCASE = True

# 理论技术重复编号
EXPECTED_TECH_REPEATS = {1, 2, 3}

# ============================================================
# PLSR + Hotelling T2 + Q residual 异常光谱剔除设置
# ============================================================
# True = 在所有单次测量光谱中建立 PLSR，并在求均值前剔除异常单条光谱
ENABLE_PLSR_T2_Q_FILTER = True

# 置信水平：95%
PLSR_CONFIDENCE_LEVEL = 0.95

# PLSR 最大潜变量数。脚本会用交叉验证在 1 ~ 该上限内自动选择。
PLSR_MAX_COMPONENTS = 10

# 样本量太小时不做剔除，避免阈值不稳定。
PLSR_MIN_SAMPLES_FOR_FILTER = 10

# 剔除规则：
# "OR"  = 只要超过 Hotelling T2 阈值 或 Q residual 阈值，就剔除整条单次测量光谱。
# "AND" = 只有同时超过 Hotelling T2 阈值 和 Q residual 阈值，才剔除整条单次测量光谱。
PLSR_OUTLIER_RULE = "OR"

# Q residual 阈值默认用经验 95% 分位数。
# T2 阈值优先用 F 分布 95% 限；如果 scipy 不可用，则退回经验 95% 分位数。
PLSR_Q_LIMIT_METHOD = "empirical_quantile_95"
PLSR_T2_LIMIT_METHOD = "f_distribution_95_if_available"



# ============================================================
# 2. 文件名解析
# ============================================================
def parse_cj_filename(file_path: Path):
    """
    解析文件名：
    1) 掺假样本：cj_F9;2;3_20260523_115209.001
       batch_id     = F9
       encounter_id = 2
       tech_rep_id  = 3
       date         = 20260523
       time         = 115209

    2) 纯木薯淀粉参考：cj_mu;7_20260524_132610.001
       当 INCLUDE_MU_REFERENCE=True 时，记为 batch_id=MU, encounter_id=1,
       tech_rep_id=7, adulteration_percent=100。
    """
    stem = file_path.stem

    pattern_f = (
        r"^cj_(?P<batch>[Ff]\d+);"
        r"(?P<encounter>\d+);"
        r"(?P<tech>\d+)_"
        r"(?P<date>\d{8})_"
        r"(?P<time>\d{6})$"
    )
    match_f = re.match(pattern_f, stem)

    if match_f:
        batch_id = match_f.group("batch")
        if NORMALIZE_F_TO_UPPERCASE:
            batch_id = batch_id.upper()

        encounter_id = int(match_f.group("encounter"))
        tech_rep_id = int(match_f.group("tech"))
        date = match_f.group("date")
        time = match_f.group("time")
        sample_kind = "F_sample"
        return batch_id, encounter_id, tech_rep_id, date, time, sample_kind

    pattern_mu = (
        r"^cj_[Mm][Uu];"
        r"(?P<tech>\d+)_"
        r"(?P<date>\d{8})_"
        r"(?P<time>\d{6})$"
    )
    match_mu = re.match(pattern_mu, stem)

    if match_mu and INCLUDE_MU_REFERENCE:
        batch_id = "MU"
        encounter_id = 1
        tech_rep_id = int(match_mu.group("tech"))
        date = match_mu.group("date")
        time = match_mu.group("time")
        sample_kind = "MU_reference_100pct"
        return batch_id, encounter_id, tech_rep_id, date, time, sample_kind

    raise ValueError(f"文件名不符合 cj_Fx;x;x 或启用后的 cj_mu;x 规则：{file_path.name}")

# ============================================================
# 3. 读取掺假比例表
# ============================================================
def clean_colname(x):
    return str(x).strip().replace(" ", "")


def normalize_batch_id(x):
    x = str(x).strip()
    if NORMALIZE_F_TO_UPPERCASE:
        x = x.upper()
    return x


def ratio_to_percent(x):
    """
    Excel 里如果写 0.16，输出 16%；
    如果已经写 16，则输出 16%。
    """
    if pd.isna(x):
        return np.nan
    val = float(x)
    if abs(val) <= 1:
        return val * 100
    return val


def load_ratio_table(table_path: Path):
    if not table_path.exists():
        raise FileNotFoundError(f"找不到掺假比例表：{table_path}")

    df = pd.read_excel(table_path)
    df.columns = [clean_colname(c) for c in df.columns]

    if "编号" not in df.columns:
        raise ValueError(f"Excel 必须有 '编号' 列。当前列名：{list(df.columns)}")

    # 自动识别“第1次比例”到“第9次比例”等列。
    # 兼容：第1次比例 / 第一次比例 / 1比例 / 一比例。
    cn_num = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9,
    }

    ratio_cols = {}
    for col in df.columns:
        col_str = str(col).strip().replace(" ", "")
        m = re.match(r"^第?(\d+)次?比例$", col_str)
        if m:
            ratio_cols[int(m.group(1))] = col
            continue
        m = re.match(r"^第?([一二三四五六七八九])次?比例$", col_str)
        if m:
            ratio_cols[cn_num[m.group(1)]] = col
            continue

    if not ratio_cols:
        raise ValueError(
            "Excel 中未识别到比例列。请使用类似 '第1次比例'、'第2次比例' 的列名。"
            f" 当前列名：{list(df.columns)}"
        )

    ratio_cols = dict(sorted(ratio_cols.items(), key=lambda x: x[0]))

    ratio_records = []
    for _, row in df.iterrows():
        batch_id = normalize_batch_id(row["编号"])
        for encounter_id, col in ratio_cols.items():
            ratio_raw = row[col]
            if pd.isna(ratio_raw):
                continue
            ratio_records.append({
                "batch_id": batch_id,
                "encounter_id": int(encounter_id),
                "adulteration_ratio": float(ratio_raw),
                "adulteration_percent": ratio_to_percent(ratio_raw),
                "ratio_source_column": col,
            })

    ratio_df = pd.DataFrame(ratio_records)

    if ratio_df.empty:
        raise ValueError("比例表识别成功，但所有比例值均为空。")

    # 可选加入纯木薯淀粉参考：用于输出 100% 光谱参考，不建议默认参与 F 样品模型。
    if INCLUDE_MU_REFERENCE:
        ratio_df = pd.concat([
            ratio_df,
            pd.DataFrame([{
                "batch_id": "MU",
                "encounter_id": 1,
                "adulteration_ratio": 1.0,
                "adulteration_percent": 100.0,
                "ratio_source_column": "MU_reference_100pct",
            }])
        ], ignore_index=True)

    duplicated = ratio_df.duplicated(["batch_id", "encounter_id"], keep=False)
    if duplicated.any():
        dup = ratio_df.loc[duplicated, ["batch_id", "encounter_id", "adulteration_ratio"]]
        raise ValueError("Excel 比例表存在重复 batch_id + encounter_id：\n" + dup.to_string(index=False))

    return ratio_df

# ============================================================
# 4. 从 OPUS 参数生成波数轴
# ============================================================
def get_axis_from_params(params, n_points):
    """
    从 OPUS Data Parameter 里生成波数轴。
    常用参数：
    FXV = first x value
    LXV = last x value
    """
    x_start = float(params["FXV"])
    x_end = float(params["LXV"])
    return np.linspace(x_start, x_end, n_points)


# ============================================================
# 5. 插值函数：把 ScRf 对齐到 ScSm 波数轴
# ============================================================
def interpolate_to_axis(x_source, y_source, x_target):
    """
    np.interp 需要 x 升序。
    OPUS 波数轴常常是从高到低，所以这里自动处理升降序。
    """
    x_source = np.asarray(x_source, dtype=float)
    y_source = np.asarray(y_source, dtype=float)
    x_target = np.asarray(x_target, dtype=float)

    if x_source[0] > x_source[-1]:
        x_source_sorted = x_source[::-1]
        y_source_sorted = y_source[::-1]
    else:
        x_source_sorted = x_source
        y_source_sorted = y_source

    if x_target[0] > x_target[-1]:
        x_target_sorted = x_target[::-1]
        reverse_back = True
    else:
        x_target_sorted = x_target
        reverse_back = False

    y_interp = np.interp(
        x_target_sorted,
        x_source_sorted,
        y_source_sorted,
        left=np.nan,
        right=np.nan,
    )

    if reverse_back:
        y_interp = y_interp[::-1]

    return y_interp


# ============================================================
# 6. 读取 OPUS 光谱：ScSm / ScRf -> Abs
# ============================================================
def read_opus_spectrum(file_path: Path):
    opus = read_file(str(file_path))

    required_keys = [
        "ScSm",
        "ScRf",
        "ScSm Data Parameter",
        "ScRf Data Parameter",
    ]

    missing_keys = [k for k in required_keys if k not in opus]
    if missing_keys:
        raise ValueError(
            f"{file_path.name} 缺少必要数据块：{missing_keys}。"
            f"可用 keys 为：{list(opus.keys())}"
        )

    scsm = np.asarray(opus["ScSm"], dtype=float)
    scrf = np.asarray(opus["ScRf"], dtype=float)

    x_sm = get_axis_from_params(opus["ScSm Data Parameter"], len(scsm))
    x_rf = get_axis_from_params(opus["ScRf Data Parameter"], len(scrf))

    x = x_sm
    scrf_interp = interpolate_to_axis(x_rf, scrf, x_sm)

    eps = 1e-12
    scsm_safe = np.where(scsm > eps, scsm, np.nan)
    scrf_safe = np.where(scrf_interp > eps, scrf_interp, np.nan)

    absorbance = -np.log10(scsm_safe / scrf_safe)

    valid = np.isfinite(absorbance)
    x = x[valid]
    absorbance = absorbance[valid]

    if len(absorbance) < 100:
        raise ValueError(f"{file_path.name} 计算出的有效吸光度点太少。")

    abs_min = float(np.nanmin(absorbance))
    abs_max = float(np.nanmax(absorbance))

    if np.nanmax(np.abs(absorbance)) > 20:
        raise ValueError(
            f"{file_path.name} 计算出的吸光度范围异常："
            f"min={abs_min}, max={abs_max}。请检查 ScSm/ScRf。"
        )

    return x, absorbance, "Abs_from_ScSm_ScRf"


# ============================================================
# 7. 光谱轴对齐
# ============================================================
def align_to_reference(x, y, x_ref):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_ref = np.asarray(x_ref, dtype=float)

    if len(x) == len(x_ref) and np.allclose(x, x_ref, rtol=0, atol=1e-6):
        return y

    return interpolate_to_axis(x, y, x_ref)


# ============================================================
# 8. 安全文件名
# ============================================================
def safe_filename(name: str):
    return re.sub(r'[\\/:*?"<>|]', "_", str(name))



# ============================================================
# 9. PLSR + Hotelling T2 + Q residual 异常光谱剔除
# ============================================================
def select_plsr_n_components_by_cv(X, y, max_components=10, random_state=42):
    """
    用交叉验证选择 PLSR 潜变量数。
    只用于异常检测前的全局 PLSR 建模，不改变原始光谱。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    n_samples, n_features = X.shape
    if n_samples < 3:
        return 1, pd.DataFrame([{
            "n_components": 1,
            "cv_rmse": np.nan,
            "note": "too_few_samples"
        }])

    n_splits = min(5, n_samples)
    # KFold 中最小训练集大小约为 n - ceil(n / n_splits)
    min_train_size = n_samples - int(np.ceil(n_samples / n_splits))

    max_allowed = min(
        int(max_components),
        int(n_features),
        int(max(1, min_train_size - 1))
    )

    if max_allowed < 1:
        return 1, pd.DataFrame([{
            "n_components": 1,
            "cv_rmse": np.nan,
            "note": "max_allowed_less_than_1"
        }])

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_records = []
    best_a = 1
    best_rmse = np.inf

    for a in range(1, max_allowed + 1):
        y_pred_all = np.full_like(y, fill_value=np.nan, dtype=float)

        valid_fold = True
        for train_idx, test_idx in kf.split(X):
            if len(train_idx) <= a:
                valid_fold = False
                break

            model = PLSRegression(n_components=a, scale=True)
            model.fit(X[train_idx], y[train_idx])
            y_pred_all[test_idx, 0] = model.predict(X[test_idx]).ravel()

        if not valid_fold or np.isnan(y_pred_all).any():
            rmse = np.nan
        else:
            rmse = float(np.sqrt(mean_squared_error(y.ravel(), y_pred_all.ravel())))

        cv_records.append({
            "n_components": a,
            "cv_rmse": rmse,
            "note": ""
        })

        if np.isfinite(rmse) and rmse < best_rmse:
            best_rmse = rmse
            best_a = a

    return best_a, pd.DataFrame(cv_records)


def _get_pls_x_center_scale(pls):
    """
    兼容不同 sklearn 版本中 PLSRegression 的 X 均值/标准差属性名。
    """
    x_mean = None
    x_std = None

    for attr in ["_x_mean", "x_mean_"]:
        if hasattr(pls, attr):
            x_mean = getattr(pls, attr)
            break

    for attr in ["_x_std", "x_std_"]:
        if hasattr(pls, attr):
            x_std = getattr(pls, attr)
            break

    if x_mean is None:
        # 理论上不会走到这里；只是兜底，避免不同版本属性名导致脚本中断。
        x_mean = np.zeros(pls.x_loadings_.shape[0], dtype=float)
    if x_std is None:
        x_std = np.ones_like(x_mean, dtype=float)

    x_std = np.asarray(x_std, dtype=float)
    x_std = np.where(np.abs(x_std) < 1e-12, 1.0, x_std)

    return np.asarray(x_mean, dtype=float), x_std


def plsr_hotelling_t2_q_filter(X, y, confidence=0.95, max_components=10):
    """
    在所有单次测量光谱上建立 PLSR，并计算每条光谱的：
    - Hotelling T2：基于 PLS X-scores
    - Q residual：基于标准化 X 空间中的 X 残差平方和

    本函数只判断每条光谱是否超过 T2 阈值、是否超过 Q 阈值。
    后面用 AND 规则决定是否剔除整条单次测量光谱。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    valid_y = np.isfinite(y)
    if valid_y.sum() < PLSR_MIN_SAMPLES_FOR_FILTER:
        diag = pd.DataFrame({
            "plsr_available_for_filter": False,
            "reason": f"finite_y_samples_less_than_{PLSR_MIN_SAMPLES_FOR_FILTER}",
            "selected_n_components": np.nan,
            "T2": np.nan,
            "Q_residual": np.nan,
            "T2_limit_95": np.nan,
            "Q_limit_95": np.nan,
            "T2_limit_method": "",
            "Q_limit_method": "",
            "T2_over_limit": False,
            "Q_over_limit": False,
        }, index=np.arange(X.shape[0]))
        cv_df = pd.DataFrame()
        return diag, cv_df

    X_fit = X[valid_y]
    y_fit = y[valid_y]

    selected_a, cv_df = select_plsr_n_components_by_cv(
        X_fit,
        y_fit,
        max_components=max_components
    )

    pls = PLSRegression(n_components=int(selected_a), scale=True)
    pls.fit(X_fit, y_fit.reshape(-1, 1))

    # 用训练时的 X 均值和标准差，把所有光谱放到同一标准化空间。
    x_mean, x_std = _get_pls_x_center_scale(pls)
    X_scaled_all = (X - x_mean) / x_std

    # PLS 分数：对所有样本投影。
    T_all = pls.transform(X)
    if isinstance(T_all, tuple):
        T_all = T_all[0]
    T_all = np.asarray(T_all, dtype=float)

    # Q residual：X_scaled - T P'
    P = np.asarray(pls.x_loadings_, dtype=float)
    X_hat_scaled = T_all @ P.T
    E = X_scaled_all - X_hat_scaled
    Q = np.sum(E ** 2, axis=1)

    # Hotelling T2：score covariance 的二次型。这里用训练集 score 方差。
    T_fit = T_all[valid_y]
    score_var = np.var(T_fit, axis=0, ddof=1)
    score_var = np.where(score_var <= 1e-12, np.nan, score_var)
    T2 = np.nansum((T_all ** 2) / score_var, axis=1)

    n_fit = int(valid_y.sum())
    a = int(selected_a)

    # T2 95% 阈值：优先 F 分布；否则经验 95% 分位数兜底。
    if SCIPY_AVAILABLE and n_fit > a:
        T2_limit = float(
            a * (n_fit - 1) / (n_fit - a)
            * f_distribution.ppf(confidence, a, n_fit - a)
        )
        T2_limit_method = "F_distribution"
    else:
        T2_limit = float(np.nanquantile(T2[valid_y], confidence))
        T2_limit_method = "empirical_quantile"

    # Q residual 95% 阈值：经验 95% 分位数。
    Q_limit = float(np.nanquantile(Q[valid_y], confidence))
    Q_limit_method = "empirical_quantile"

    diag = pd.DataFrame({
        "plsr_available_for_filter": True,
        "reason": "",
        "selected_n_components": a,
        "T2": T2,
        "Q_residual": Q,
        "T2_limit_95": T2_limit,
        "Q_limit_95": Q_limit,
        "T2_limit_method": T2_limit_method,
        "Q_limit_method": Q_limit_method,
        "T2_over_limit": T2 > T2_limit,
        "Q_over_limit": Q > Q_limit,
    })

    return diag, cv_df



# ============================================================
# 10. 主流程
# ============================================================
def main():
    print("开始读取掺假比例表...")
    ratio_df = load_ratio_table(TABLE_PATH)
    ratio_df.to_csv(
        OUTPUT_DIR / "00_ratio_mapping_from_excel.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"比例表记录数：{len(ratio_df)}")

    files = sorted(INPUT_DIR.glob("*.001"))
    if not files:
        raise FileNotFoundError(
            f"没有找到 .001 文件：{INPUT_DIR}\n"
            f"请检查 .001 文件是否放在脚本同目录下的 DATA 文件夹中，"
            f"或把 INPUT_DIR 手动改成真实数据文件夹。"
        )

    print()
    print("开始读取 OPUS 文件...")
    print(f"数据文件夹：{INPUT_DIR}")
    print(f"发现 .001 文件数：{len(files)}")

    spectra_records = []
    metadata_records = []
    skipped_records = []
    x_ref = None
    used_blocks = set()

    for file_path in files:
        try:
            batch_id, encounter_id, tech_rep_id, date, time, sample_kind = parse_cj_filename(file_path)
        except ValueError as e:
            if SKIP_NON_CJ_F_FILES:
                skipped_records.append({
                    "file_name": file_path.name,
                    "reason": str(e),
                })
                continue
            raise

        x, y, block = read_opus_spectrum(file_path)

        if x_ref is None:
            x_ref = x

        y_aligned = align_to_reference(x, y, x_ref)
        used_blocks.add(block)

        # 连接 Excel 比例
        ratio_match = ratio_df[
            (ratio_df["batch_id"] == batch_id)
            & (ratio_df["encounter_id"] == encounter_id)
        ]

        if ratio_match.empty:
            adulteration_ratio = np.nan
            adulteration_percent = np.nan
            ratio_source_column = "NOT_FOUND"
            ratio_match_status = "not_found_in_excel"
        else:
            ratio_row = ratio_match.iloc[0]
            adulteration_ratio = ratio_row["adulteration_ratio"]
            adulteration_percent = ratio_row["adulteration_percent"]
            ratio_source_column = ratio_row["ratio_source_column"]
            ratio_match_status = "matched"

        spectra_records.append({
            "file_name": file_path.name,
            "batch_id": batch_id,
            "encounter_id": encounter_id,
            "tech_rep_id": tech_rep_id,
            "sample_state_id": f"{batch_id};{encounter_id}",
            "date": date,
            "time": time,
            "opus_block": block,
            "sample_kind": sample_kind,
            "adulteration_ratio": adulteration_ratio,
            "adulteration_percent": adulteration_percent,
            "ratio_source_column": ratio_source_column,
            "ratio_match_status": ratio_match_status,
            "spectrum": y_aligned,
        })

        metadata_records.append({
            "file_name": file_path.name,
            "batch_id": batch_id,
            "encounter_id": encounter_id,
            "tech_rep_id": tech_rep_id,
            "sample_state_id": f"{batch_id};{encounter_id}",
            "date": date,
            "time": time,
            "opus_block": block,
            "sample_kind": sample_kind,
            "adulteration_ratio": adulteration_ratio,
            "adulteration_percent": adulteration_percent,
            "ratio_source_column": ratio_source_column,
            "ratio_match_status": ratio_match_status,
            "n_points_after_valid_filter": len(y),
            "wavenumber_start": float(x[0]),
            "wavenumber_end": float(x[-1]),
            "abs_min": float(np.nanmin(y)),
            "abs_max": float(np.nanmax(y)),
            "abs_mean": float(np.nanmean(y)),
            "abs_median": float(np.nanmedian(y)),
        })

    if not spectra_records:
        raise ValueError("没有成功读取任何符合 cj_Fx;x;x 规则的 F 样本文件。")

    df_meta = pd.DataFrame(metadata_records)
    df_meta = df_meta.sort_values(["batch_id", "encounter_id", "tech_rep_id", "date", "time"])
    df_meta.to_csv(
        OUTPUT_DIR / "01_file_metadata_with_ratio.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if skipped_records:
        pd.DataFrame(skipped_records).to_csv(
            OUTPUT_DIR / "01_skipped_non_cj_F_files.csv",
            index=False,
            encoding="utf-8-sig",
        )

    print(f"成功读取 F 样本光谱数：{len(spectra_records)}")
    print(f"跳过非 cj_Fx;x;x 文件数：{len(skipped_records)}")

    # ========================================================
    # 10. 统一光谱矩阵，保证所有输出表使用相同波数列
    # ========================================================
    X_all = np.vstack([r["spectrum"] for r in spectra_records]).astype(float)

    # 删除所有光谱均为 NaN 的波数点
    valid_cols = ~np.isnan(X_all).all(axis=0)
    X_valid = X_all[:, valid_cols]
    x_valid = x_ref[valid_cols]

    # 少量 NaN 用该波数点所有样本均值填补
    col_mean = np.nanmean(X_valid, axis=0)
    still_valid = np.isfinite(col_mean)
    X_valid = X_valid[:, still_valid]
    x_valid = x_valid[still_valid]
    col_mean = col_mean[still_valid]

    nan_pos = np.where(np.isnan(X_valid))
    X_valid[nan_pos] = np.take(col_mean, nan_pos[1])

    # 把清洗后的光谱放回 records，方便后续分组
    for i, r in enumerate(spectra_records):
        r["spectrum_clean"] = X_valid[i, :]

    feature_cols = [f"wn_{wn:.4f}" for wn in x_valid]

    # ========================================================
    # 11. PLSR + Hotelling T2 + Q residual 异常光谱剔除
    # 注意：这里剔除的是“整条单次测量光谱/一个 .001 文件”，不是删除波数点。
    # 剔除规则：同时超过 T2 95% 阈值 和 Q residual 95% 阈值，才剔除。
    # 被剔除的文件不参与后续均值光谱计算。
    # ========================================================
    plsr_cv_df = pd.DataFrame()

    if ENABLE_PLSR_T2_Q_FILTER:
        y_all = np.array([r["adulteration_percent"] for r in spectra_records], dtype=float)

        plsr_diag, plsr_cv_df = plsr_hotelling_t2_q_filter(
            X_valid,
            y_all,
            confidence=PLSR_CONFIDENCE_LEVEL,
            max_components=PLSR_MAX_COMPONENTS,
        )

        available_mask = plsr_diag["plsr_available_for_filter"].astype(bool).to_numpy()
        t2_over_mask = plsr_diag["T2_over_limit"].astype(bool).to_numpy()
        q_over_mask = plsr_diag["Q_over_limit"].astype(bool).to_numpy()

        rule = str(PLSR_OUTLIER_RULE).strip().upper()
        if rule == "OR":
            # 删除逻辑：只要 T2 或 Q residual 任一指标超过 95% 阈值，就剔除整条单次测量光谱。
            remove_mask = available_mask & (t2_over_mask | q_over_mask)
            filter_rule_label = "T2_over_95_OR_Q_over_95"
        elif rule == "AND":
            # 删除逻辑：只有 T2 和 Q residual 同时超过 95% 阈值，才剔除整条单次测量光谱。
            remove_mask = available_mask & t2_over_mask & q_over_mask
            filter_rule_label = "T2_over_95_AND_Q_over_95"
        else:
            raise ValueError("PLSR_OUTLIER_RULE 只能设置为 'OR' 或 'AND'。")

        for i, r in enumerate(spectra_records):
            diag_row = plsr_diag.iloc[i].to_dict()
            for k, v in diag_row.items():
                r[f"plsr_{k}"] = v

            r["plsr_filter_rule"] = filter_rule_label
            r["plsr_outlier_removed"] = bool(remove_mask[i])
            r["plsr_filter_decision"] = "remove_from_mean" if remove_mask[i] else "keep_for_mean"

    else:
        for r in spectra_records:
            r["plsr_plsr_available_for_filter"] = False
            r["plsr_reason"] = "PLSR_T2_Q_filter_disabled"
            r["plsr_selected_n_components"] = np.nan
            r["plsr_T2"] = np.nan
            r["plsr_Q_residual"] = np.nan
            r["plsr_T2_limit_95"] = np.nan
            r["plsr_Q_limit_95"] = np.nan
            r["plsr_T2_over_limit"] = False
            r["plsr_Q_over_limit"] = False
            r["plsr_filter_rule"] = "disabled"
            r["plsr_outlier_removed"] = False
            r["plsr_filter_decision"] = "keep_for_mean"

    plsr_rows = []
    for r in spectra_records:
        plsr_rows.append({
            "file_name": r["file_name"],
            "batch_id": r["batch_id"],
            "encounter_id": r["encounter_id"],
            "tech_rep_id": r["tech_rep_id"],
            "sample_state_id": r["sample_state_id"],
            "sample_kind": r.get("sample_kind", ""),
            "adulteration_ratio": r["adulteration_ratio"],
            "adulteration_percent": r["adulteration_percent"],
            "plsr_selected_n_components": r.get("plsr_selected_n_components", np.nan),
            "T2": r.get("plsr_T2", np.nan),
            "Q_residual": r.get("plsr_Q_residual", np.nan),
            "T2_limit_95": r.get("plsr_T2_limit_95", np.nan),
            "Q_limit_95": r.get("plsr_Q_limit_95", np.nan),
            "T2_limit_method": r.get("plsr_T2_limit_method", ""),
            "Q_limit_method": r.get("plsr_Q_limit_method", ""),
            "T2_over_limit": r.get("plsr_T2_over_limit", False),
            "Q_over_limit": r.get("plsr_Q_over_limit", False),
            "outlier_rule": r.get("plsr_filter_rule", ""),
            "outlier_removed": r.get("plsr_outlier_removed", False),
            "filter_decision": r.get("plsr_filter_decision", ""),
            "reason": r.get("plsr_reason", ""),
        })

    df_plsr_outlier = pd.DataFrame(plsr_rows)
    df_plsr_outlier.to_csv(
        OUTPUT_DIR / "08_plsr_T2_Q_outlier_diagnostics_all_single_spectra.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_plsr_removed = df_plsr_outlier[df_plsr_outlier["outlier_removed"] == True].copy()
    df_plsr_removed.to_csv(
        OUTPUT_DIR / "09_removed_single_spectra_by_plsr_T2_Q.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not plsr_cv_df.empty:
        plsr_cv_df.to_csv(
            OUTPUT_DIR / "10_plsr_component_selection_cv.csv",
            index=False,
            encoding="utf-8-sig",
        )

    spectra_records_for_mean = [
        r for r in spectra_records
        if not bool(r.get("plsr_outlier_removed", False))
    ]

    if not spectra_records_for_mean:
        raise ValueError("PLSR T2/Q 剔除后没有剩余光谱，无法继续计算均值。请检查阈值或原始数据。")


    # ========================================================
    # 11. 输出单条光谱建模表：一行 = 一个 .001 文件
    # ========================================================
    single_rows = []
    for r in spectra_records:
        row = {
            "file_name": r["file_name"],
            "batch_id": r["batch_id"],
            "encounter_id": r["encounter_id"],
            "tech_rep_id": r["tech_rep_id"],
            "sample_state_id": r["sample_state_id"],
            "sample_kind": r.get("sample_kind", ""),
            "adulteration_ratio": r["adulteration_ratio"],
            "adulteration_percent": r["adulteration_percent"],
            "date": r["date"],
            "time": r["time"],
            "opus_block": r["opus_block"],
            "plsr_filter_decision": r.get("plsr_filter_decision", ""),
            "plsr_outlier_removed": r.get("plsr_outlier_removed", False),
            "plsr_T2": r.get("plsr_T2", np.nan),
            "plsr_Q_residual": r.get("plsr_Q_residual", np.nan),
            "plsr_T2_limit_95": r.get("plsr_T2_limit_95", np.nan),
            "plsr_Q_limit_95": r.get("plsr_Q_limit_95", np.nan),
            "plsr_T2_over_limit": r.get("plsr_T2_over_limit", False),
            "plsr_Q_over_limit": r.get("plsr_Q_over_limit", False),
        }
        row.update(dict(zip(feature_cols, r["spectrum_clean"])))
        single_rows.append(row)

    df_single = pd.DataFrame(single_rows)
    df_single = df_single.sort_values(["batch_id", "encounter_id", "tech_rep_id", "date", "time"])
    df_single.to_csv(
        OUTPUT_DIR / "02_single_spectra_with_ratio_for_modeling.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_single_filtered = df_single[df_single["plsr_outlier_removed"] == False].copy()
    df_single_filtered.to_csv(
        OUTPUT_DIR / "02A_single_spectra_after_plsr_T2_Q_filter_for_modeling.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 12. 按 batch_id + encounter_id 计算均值光谱
    # ========================================================
    qc_records = []
    mean_rows = []
    naming_issue_records = []

    df_records = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ["spectrum", "spectrum_clean"]}
        for r in spectra_records_for_mean
    ])

    group_keys = ["batch_id", "encounter_id"]
    grouped = df_records.groupby(group_keys, sort=True)

    for (batch_id, encounter_id), g in grouped:
        idx = g.index.to_list()
        group_items = [spectra_records_for_mean[i] for i in idx]
        group_items = sorted(group_items, key=lambda z: (z["tech_rep_id"], z["date"], z["time"]))

        X_group = np.vstack([r["spectrum_clean"] for r in group_items]).astype(float)

        # 同一 batch_id + encounter_id 中，因 PLSR T2+Q 规则被剔除的单次测量光谱
        removed_in_this_group = [
            r for r in spectra_records
            if r["batch_id"] == batch_id
            and r["encounter_id"] == encounter_id
            and bool(r.get("plsr_outlier_removed", False))
        ]

        mean_spectrum = X_group.mean(axis=0)
        sd_spectrum = X_group.std(axis=0, ddof=1) if X_group.shape[0] > 1 else np.zeros(X_group.shape[1])

        tech_ids = [int(r["tech_rep_id"]) for r in group_items]
        unique_tech_ids = sorted(set(tech_ids))
        missing_tech = sorted(EXPECTED_TECH_REPEATS - set(tech_ids))
        duplicate_tech = sorted([t for t in set(tech_ids) if tech_ids.count(t) > 1])

        ratio_values = [r["adulteration_ratio"] for r in group_items]
        percent_values = [r["adulteration_percent"] for r in group_items]
        ratio_unique = sorted(set([v for v in ratio_values if not pd.isna(v)]))
        percent_unique = sorted(set([v for v in percent_values if not pd.isna(v)]))

        if len(ratio_unique) == 1:
            adulteration_ratio = ratio_unique[0]
        else:
            adulteration_ratio = np.nan

        if len(percent_unique) == 1:
            adulteration_percent = percent_unique[0]
        else:
            adulteration_percent = np.nan

        sample_state_id = f"{batch_id};{encounter_id}"

        # 每组一个 CSV：含重复、均值、SD
        group_df = pd.DataFrame({"wavenumber_cm-1": x_valid})
        for r in group_items:
            group_df[f"tech_{r['tech_rep_id']}_{r['time']}"] = r["spectrum_clean"]
        group_df["mean"] = mean_spectrum
        group_df["sd"] = sd_spectrum
        group_df["adulteration_ratio"] = adulteration_ratio
        group_df["adulteration_percent"] = adulteration_percent
        group_df["n_tech_repeats_used_after_plsr_filter"] = X_group.shape[0]
        group_df["n_removed_by_plsr_T2_Q"] = len(removed_in_this_group)

        group_csv_name = safe_filename(f"{sample_state_id}_ratio_{adulteration_percent:.2f}pct.csv")
        group_df.to_csv(
            GROUP_SPECTRA_DIR / group_csv_name,
            index=False,
            encoding="utf-8-sig",
        )

        # 均值光谱单独导出
        mean_df = pd.DataFrame({
            "wavenumber_cm-1": x_valid,
            "mean_absorbance": mean_spectrum,
            "sd_absorbance": sd_spectrum,
            "batch_id": batch_id,
            "encounter_id": encounter_id,
            "sample_state_id": sample_state_id,
            "adulteration_ratio": adulteration_ratio,
            "adulteration_percent": adulteration_percent,
        })
        mean_csv_name = safe_filename(f"MEAN_{sample_state_id}_ratio_{adulteration_percent:.2f}pct.csv")
        mean_df.to_csv(
            MEAN_SPECTRA_DIR / mean_csv_name,
            index=False,
            encoding="utf-8-sig",
        )

        # 建模用均值宽表：一行 = 一个 F批号 + 第几次遇见
        mean_row = {
            "sample_state_id": sample_state_id,
            "batch_id": batch_id,
            "encounter_id": encounter_id,
            "adulteration_ratio": adulteration_ratio,
            "adulteration_percent": adulteration_percent,
            "n_tech_repeats_used_after_plsr_filter": len(tech_ids),
            "n_removed_by_plsr_T2_Q": len(removed_in_this_group),
            "removed_tech_rep_ids_by_plsr_T2_Q": ",".join(map(str, [int(r["tech_rep_id"]) for r in removed_in_this_group])),
            "removed_files_by_plsr_T2_Q": " | ".join([r["file_name"] for r in removed_in_this_group]),
            "tech_rep_ids": ",".join(map(str, tech_ids)),
            "missing_tech_rep_ids": ",".join(map(str, missing_tech)) if missing_tech else "",
            "duplicate_tech_rep_ids": ",".join(map(str, duplicate_tech)) if duplicate_tech else "",
            "source_files": " | ".join([r["file_name"] for r in group_items]),
        }
        mean_row.update(dict(zip(feature_cols, mean_spectrum)))
        mean_rows.append(mean_row)

        # QC 指标
        rmsd_each = np.sqrt(np.mean((X_group - mean_spectrum) ** 2, axis=1))
        pairwise_rmsd = []
        if X_group.shape[0] > 1:
            for i in range(X_group.shape[0]):
                for j in range(i + 1, X_group.shape[0]):
                    pairwise_rmsd.append(np.sqrt(np.mean((X_group[i] - X_group[j]) ** 2)))

        eps = 1e-12
        cv_spectrum = sd_spectrum / (np.abs(mean_spectrum) + eps)

        qc_records.append({
            "sample_state_id": sample_state_id,
            "batch_id": batch_id,
            "encounter_id": encounter_id,
            "adulteration_ratio": adulteration_ratio,
            "adulteration_percent": adulteration_percent,
            "n_tech_repeats_used_after_plsr_filter": len(tech_ids),
            "n_removed_by_plsr_T2_Q": len(removed_in_this_group),
            "removed_tech_rep_ids_by_plsr_T2_Q": ",".join(map(str, [int(r["tech_rep_id"]) for r in removed_in_this_group])),
            "removed_files_by_plsr_T2_Q": " | ".join([r["file_name"] for r in removed_in_this_group]),
            "tech_rep_ids": ",".join(map(str, tech_ids)),
            "unique_tech_rep_ids": ",".join(map(str, unique_tech_ids)),
            "missing_tech_rep_ids": ",".join(map(str, missing_tech)) if missing_tech else "",
            "duplicate_tech_rep_ids": ",".join(map(str, duplicate_tech)) if duplicate_tech else "",
            "mean_RMSD_to_group_mean": float(np.mean(rmsd_each)),
            "max_RMSD_to_group_mean": float(np.max(rmsd_each)),
            "mean_pairwise_RMSD": float(np.mean(pairwise_rmsd)) if pairwise_rmsd else np.nan,
            "max_pairwise_RMSD": float(np.max(pairwise_rmsd)) if pairwise_rmsd else np.nan,
            "mean_SD_across_wavenumbers": float(np.mean(sd_spectrum)),
            "max_SD_across_wavenumbers": float(np.max(sd_spectrum)),
            "mean_CV_percent_across_wavenumbers": float(np.mean(cv_spectrum) * 100),
            "median_CV_percent_across_wavenumbers": float(np.median(cv_spectrum) * 100),
            "max_CV_percent_across_wavenumbers": float(np.max(cv_spectrum) * 100),
            "mean_absorbance": float(np.mean(mean_spectrum)),
            "median_absorbance": float(np.median(mean_spectrum)),
            "min_absorbance": float(np.min(mean_spectrum)),
            "max_absorbance": float(np.max(mean_spectrum)),
            "n_valid_wavenumbers": int(len(x_valid)),
            "wavenumber_start": float(x_valid[0]),
            "wavenumber_end": float(x_valid[-1]),
            "group_csv": str(Path("group_spectra_with_repeats") / group_csv_name),
            "mean_csv": str(Path("mean_spectra_by_batch_encounter") / mean_csv_name),
            "source_files": " | ".join([r["file_name"] for r in group_items]),
        })

        # 命名问题记录
        if missing_tech or duplicate_tech or len(tech_ids) != 3:
            naming_issue_records.append({
                "sample_state_id": sample_state_id,
                "batch_id": batch_id,
                "encounter_id": encounter_id,
                "n_files": len(tech_ids),
                "tech_rep_ids": ",".join(map(str, tech_ids)),
                "missing_tech_rep_ids": ",".join(map(str, missing_tech)) if missing_tech else "",
                "duplicate_tech_rep_ids": ",".join(map(str, duplicate_tech)) if duplicate_tech else "",
                "issue_type": ";".join(
                    [
                        x for x in [
                            "missing" if missing_tech else "",
                            "duplicate" if duplicate_tech else "",
                            "n_not_equal_3" if len(tech_ids) != 3 else "",
                        ] if x
                    ]
                ),
                "source_files": " | ".join([r["file_name"] for r in group_items]),
            })

    # ========================================================
    # 13. 输出均值建模表、QC、命名检查
    # ========================================================
    df_mean = pd.DataFrame(mean_rows)
    df_mean = df_mean.sort_values(["batch_id", "encounter_id"])
    df_mean.to_csv(
        OUTPUT_DIR / "03_mean_spectra_by_batch_encounter_with_ratio_for_modeling.csv",
        index=False,
        encoding="utf-8-sig",
    )

    df_qc = pd.DataFrame(qc_records)
    df_qc = df_qc.sort_values(["batch_id", "encounter_id"])
    df_qc.to_csv(
        OUTPUT_DIR / "04_repeatability_QC_by_batch_encounter.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if naming_issue_records:
        df_issue = pd.DataFrame(naming_issue_records)
    else:
        df_issue = pd.DataFrame(columns=[
            "sample_state_id", "batch_id", "encounter_id", "n_files",
            "tech_rep_ids", "missing_tech_rep_ids", "duplicate_tech_rep_ids",
            "issue_type", "source_files",
        ])

    df_issue = df_issue.sort_values(["batch_id", "encounter_id"])
    df_issue.to_csv(
        OUTPUT_DIR / "05_naming_issue_check.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 14. 检查 Excel 有、光谱没有；光谱有、Excel 没有
    # ========================================================
    observed_states = pd.DataFrame([
        {"batch_id": r["batch_id"], "encounter_id": r["encounter_id"]}
        for r in spectra_records
    ]).drop_duplicates()
    expected_states = ratio_df[["batch_id", "encounter_id"]].drop_duplicates()

    missing_spectra = expected_states.merge(
        observed_states,
        on=["batch_id", "encounter_id"],
        how="left",
        indicator=True,
    )
    missing_spectra = missing_spectra[missing_spectra["_merge"] == "left_only"].drop(columns="_merge")
    missing_spectra.to_csv(
        OUTPUT_DIR / "06_excel_has_ratio_but_no_spectrum.csv",
        index=False,
        encoding="utf-8-sig",
    )

    spectrum_no_ratio = observed_states.merge(
        expected_states,
        on=["batch_id", "encounter_id"],
        how="left",
        indicator=True,
    )
    spectrum_no_ratio = spectrum_no_ratio[spectrum_no_ratio["_merge"] == "left_only"].drop(columns="_merge")
    spectrum_no_ratio.to_csv(
        OUTPUT_DIR / "07_spectrum_has_no_ratio_in_excel.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 15. 终端总结
    # ========================================================
    print()
    print("处理完成。")
    print("=" * 70)
    print(f"读取 .001 文件总数：{len(files)}")
    print(f"成功读取 F 样本光谱数：{len(spectra_records)}")
    print(f"跳过非 cj_Fx;x;x 文件数：{len(skipped_records)}")
    print(f"PLSR T2+Q 剔除单次测量光谱数：{int(df_plsr_removed.shape[0])}")
    print(f"进入均值计算的单次测量光谱数：{len(spectra_records_for_mean)}")
    print(f"样本状态数 batch_id + encounter_id：{df_mean.shape[0]}")
    print(f"使用光谱类型：{used_blocks}")
    print(f"参考波数点数量：{len(x_ref)}")
    print(f"清洗后波数点数量：{len(x_valid)}")
    print(f"输出目录：{OUTPUT_DIR}")
    print()
    print("主要输出文件：")
    print("00_ratio_mapping_from_excel.csv")
    print("01_file_metadata_with_ratio.csv")
    print("02_single_spectra_with_ratio_for_modeling.csv")
    print("02A_single_spectra_after_plsr_T2_Q_filter_for_modeling.csv")
    print("03_mean_spectra_by_batch_encounter_with_ratio_for_modeling.csv")
    print("04_repeatability_QC_by_batch_encounter.csv")
    print("05_naming_issue_check.csv")
    print("06_excel_has_ratio_but_no_spectrum.csv")
    print("07_spectrum_has_no_ratio_in_excel.csv")
    print("08_plsr_T2_Q_outlier_diagnostics_all_single_spectra.csv")
    print("09_removed_single_spectra_by_plsr_T2_Q.csv")
    print("10_plsr_component_selection_cv.csv")
    print("mean_spectra_by_batch_encounter/ 每个样本状态一个均值光谱 CSV")
    print("group_spectra_with_repeats/ 每个样本状态一个重复+均值+SD CSV")
    print("=" * 70)

    if not df_issue.empty:
        print()
        print("发现命名/重复数量问题，请优先检查 05_naming_issue_check.csv：")
        print(df_issue[[
            "sample_state_id", "tech_rep_ids", "missing_tech_rep_ids",
            "duplicate_tech_rep_ids", "issue_type",
        ]].to_string(index=False))

    if not missing_spectra.empty:
        print()
        print("Excel 有比例但没有对应光谱：")
        print(missing_spectra.to_string(index=False))

    if not spectrum_no_ratio.empty:
        print()
        print("光谱存在但 Excel 没有对应比例：")
        print(spectrum_no_ratio.to_string(index=False))


if __name__ == "__main__":
    main()
