import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PREFIX = "fingertip_position_log_a"
OUTPUT_DIR = "alpha_fingertip_analysis_figures"
DPI = 300

TARGET_ALPHAS = [round(i / 10, 1) for i in range(1, 11)]
EVAL_START_TIME = 0.5
EVAL_FINGERS = ["index", "middle", "ring", "pinky"]
DISPLAY_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
REP_FINGER = "index"
AXES = ["x", "y", "z"]

SELECTED_ALPHA_OVERRIDE = None

RMSE_WEIGHT = 0.5
DELAY_WEIGHT = 0.5
MAX_DELAY_S = 4.0


def alpha_to_name(alpha):
    return f"{int(round(float(alpha) * 10)):02d}"


def file_for_alpha(alpha):
    return f"{CSV_PREFIX}{alpha_to_name(alpha)}.csv"


def require_columns(df, cols, desc="data"):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{desc}에 필요한 열이 없습니다: {missing}")


def load_logs():
    logs = {}
    missing = []

    for alpha in TARGET_ALPHAS:
        path = file_for_alpha(alpha)

        if not os.path.exists(path):
            missing.append(path)
            continue

        df = pd.read_csv(path).dropna().copy()
        require_columns(df, ["time_s", "alpha"], path)

        df = df[df["time_s"] >= EVAL_START_TIME].copy()
        if df.empty:
            raise RuntimeError(f"{path}는 {EVAL_START_TIME}s 이후 데이터가 없습니다.")

        df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
        logs[alpha] = df.reset_index(drop=True)

    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            "다음 CSV 파일이 없습니다. 먼저 02_replay.py를 실행하세요:\n" + msg
        )

    return dict(sorted(logs.items()))


def rmse(x):
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x ** 2)))


def normalize_metric(values):
    values = np.asarray(values, dtype=float)
    vmin = float(np.min(values))
    vmax = float(np.max(values))

    if abs(vmax - vmin) < 1e-12:
        return np.zeros_like(values, dtype=float)

    return (values - vmin) / (vmax - vmin)


def savefig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=DPI)
    plt.close()
    print("saved:", path)


def style_time_plot(title, ylabel):
    plt.title(title, fontsize=14, pad=10)
    plt.xlabel("Time [s]", fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linewidth=0.35, alpha=0.35)
    plt.tight_layout()


def fingertip_displacement(df, prefix, finger):
    cols = [f"{prefix}_{finger}_{axis}" for axis in AXES]
    require_columns(df, cols, f"{prefix}-{finger}")

    pos = df[cols].to_numpy(dtype=float)
    pos0 = pos[0]

    return np.linalg.norm(pos - pos0, axis=1)


def mean_fingertip_displacement(df, prefix):
    disps = []

    for finger in EVAL_FINGERS:
        disps.append(fingertip_displacement(df, prefix, finger))

    return np.mean(np.vstack(disps), axis=0)


def estimate_tracking_delay(df):
    t = df["time_s"].to_numpy(dtype=float)
    human = mean_fingertip_displacement(df, "human")
    robot = mean_fingertip_displacement(df, "robot")

    if len(t) < 5:
        return 0.0

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return 0.0

    t_uniform = np.arange(t[0], t[-1], dt)
    if len(t_uniform) < 5:
        return 0.0

    human_i = np.interp(t_uniform, t, human)
    robot_i = np.interp(t_uniform, t, robot)

    human_i = human_i - np.mean(human_i)
    robot_i = robot_i - np.mean(robot_i)

    human_std = np.std(human_i)
    robot_std = np.std(robot_i)

    if human_std < 1e-12 or robot_std < 1e-12:
        return 0.0

    human_i = human_i / human_std
    robot_i = robot_i / robot_std

    corr = np.correlate(human_i, robot_i, mode="full")
    lags = np.arange(-len(human_i) + 1, len(human_i))

    max_lag_samples = int(MAX_DELAY_S / dt)
    valid = np.abs(lags) <= max_lag_samples

    corr = corr[valid]
    lags = lags[valid]

    best_lag = lags[np.argmax(corr)]
    delay_s = abs(float(best_lag * dt))

    return delay_s


def calc_summary(logs):
    rows = []

    for alpha, df in logs.items():
        error_cols = [f"{finger}_tip_error" for finger in EVAL_FINGERS]
        require_columns(df, error_cols, f"alpha={alpha}")

        errors = df[error_cols].to_numpy(dtype=float)
        delay_s = estimate_tracking_delay(df)

        row = {
            "alpha": alpha,
            "rmse_all": rmse(errors.reshape(-1)),
            "mean_error_all": float(np.mean(errors)),
            "max_error_all": float(np.max(errors)),
            "delay_s": delay_s,
        }

        for finger in EVAL_FINGERS:
            e = df[f"{finger}_tip_error"].to_numpy(dtype=float)
            row[f"rmse_{finger}"] = rmse(e)

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

    summary["rmse_norm"] = normalize_metric(summary["rmse_all"].to_numpy(dtype=float))
    summary["delay_norm"] = normalize_metric(summary["delay_s"].to_numpy(dtype=float))

    summary["score"] = (
        RMSE_WEIGHT * summary["rmse_norm"]
        + DELAY_WEIGHT * summary["delay_norm"]
    )

    return summary


def select_alpha(summary):
    if SELECTED_ALPHA_OVERRIDE is not None:
        if SELECTED_ALPHA_OVERRIDE not in summary["alpha"].values:
            raise ValueError(f"SELECTED_ALPHA_OVERRIDE={SELECTED_ALPHA_OVERRIDE}에 해당하는 alpha 결과가 없습니다.")
        return float(SELECTED_ALPHA_OVERRIDE)

    idx = summary["score"].idxmin()
    return float(summary.loc[idx, "alpha"])


def plot_human_input_trajectory(logs):
    source_alpha = 1.0 if 1.0 in logs else list(logs.keys())[0]
    df = logs[source_alpha]

    plt.figure(figsize=(11, 5))

    for finger in DISPLAY_FINGERS:
        disp = fingertip_displacement(df, "human", finger)
        plt.plot(df["time_s"], disp, linewidth=1.6, label=finger)

    style_time_plot("Input trajectory", "Displacement")
    plt.legend(ncol=2, fontsize=9)
    savefig("01_input_trajectory.png")


def plot_tracking_comparison(logs, selected_alpha):
    df = logs[selected_alpha]

    human_disp = fingertip_displacement(df, "human", REP_FINGER)
    robot_disp = fingertip_displacement(df, "robot", REP_FINGER)

    plt.figure(figsize=(11, 5))

    plt.plot(
        df["time_s"],
        human_disp,
        linewidth=2.2,
        label=f"human {REP_FINGER}"
    )

    plt.plot(
        df["time_s"],
        robot_disp,
        linewidth=2.0,
        label=f"robot alpha={selected_alpha}"
    )

    style_time_plot("Tracking comparison", "Displacement")
    plt.legend(loc="upper right")
    savefig("02_tracking_comparison.png")


def plot_error_over_time(logs, selected_alpha):
    df = logs[selected_alpha]
    plt.figure(figsize=(11, 5))

    for finger in DISPLAY_FINGERS:
        plt.plot(
            df["time_s"],
            df[f"{finger}_tip_error"],
            linewidth=1.1,
            label=finger
        )

    mean_eval_error = df[[f"{f}_tip_error" for f in DISPLAY_FINGERS]].mean(axis=1)
    plt.plot(
        df["time_s"],
        mean_eval_error,
        linewidth=2.7,
        label="mean error"
    )

    style_time_plot("Fingertip error", "Error")
    plt.legend(ncol=3, fontsize=9)
    savefig("03_fingertip_error.png")


def plot_alpha_selection(summary, selected_alpha):
    plt.figure(figsize=(9, 5))

    plt.plot(
        summary["alpha"],
        summary["rmse_norm"],
        marker="o",
        linewidth=1.8,
        label="RMSE"
    )

    plt.plot(
        summary["alpha"],
        summary["delay_norm"],
        marker="o",
        linewidth=1.8,
        label="delay"
    )

    plt.plot(
        summary["alpha"],
        summary["score"],
        marker="o",
        linewidth=2.4,
        label="score"
    )

    selected_row = summary[summary["alpha"] == selected_alpha].iloc[0]
    plt.scatter(
        [selected_alpha],
        [selected_row["score"]],
        s=90,
        zorder=5,
        label=f"selected alpha={selected_alpha}"
    )

    plt.title("Alpha selection", fontsize=14, pad=10)
    plt.xlabel("Alpha", fontsize=11)
    plt.ylabel("Normalized value", fontsize=11)
    plt.grid(True, linewidth=0.35, alpha=0.35)
    plt.legend(loc="best")
    plt.tight_layout()
    savefig("04_alpha_selection.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logs = load_logs()
    summary = calc_summary(logs)
    selected_alpha = select_alpha(summary)

    summary_path = os.path.join(OUTPUT_DIR, "alpha_selection_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\nTime-sync alpha selection summary")
    print(summary[["alpha", "rmse_all", "delay_s", "rmse_norm", "delay_norm", "score"]])
    print("\nSelected alpha =", selected_alpha)
    print("saved:", summary_path)

    plot_human_input_trajectory(logs)
    plot_tracking_comparison(logs, selected_alpha)
    plot_error_over_time(logs, selected_alpha)
    plot_alpha_selection(summary, selected_alpha)

    print("\n완료.")
    print(f"그래프 폴더: {OUTPUT_DIR}")
    print("생성 그래프: 4개")


if __name__ == "__main__":
    main()
