import math
from dataclasses import dataclass

import mujoco
import numpy as np
import pandas as pd


# ============================================================
# 파일 / 모델 설정
# ============================================================

MUJOCO_MODEL_PATH = r"C:\hand_mujoco_project\scene_right.xml"
RAW_LANDMARK_CSV = "raw_hand_landmarks.csv"


# filtered(t) = (1 - alpha) * filtered(t-1) + alpha * raw(t)
ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

# 초기 기준 자세가 잡히는 앞부분 제거 시간
EVAL_START_TIME = 0.5

# alpha 최적화에 사용할 손가락
# 엄지는 구조와 움직임 방향이 달라서 기본 선정 기준에서는 제외.
EVAL_FINGERS = ["index", "middle", "ring", "pinky"]


# ============================================================
# 손 / 로봇 설정
# ============================================================

FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

FINGER_LANDMARKS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

FINGERTIP_LANDMARKS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

ROBOT_TIP_CANDIDATES = {
    "thumb": ["rh_thtip", "rh_THtip", "thtip", "THtip", "rh_thdistal", "thdistal"],
    "index": ["rh_fftip", "rh_FFtip", "fftip", "FFtip", "rh_ffdistal", "ffdistal"],
    "middle": ["rh_mftip", "rh_MFtip", "mftip", "MFtip", "rh_mfdistal", "mfdistal"],
    "ring": ["rh_rftip", "rh_RFtip", "rftip", "RFtip", "rh_rfdistal", "rfdistal"],
    "pinky": ["rh_lftip", "rh_LFtip", "lftip", "LFtip", "rh_lfdistal", "lfdistal"],
}

ROBOT_PALM_CANDIDATES = [
    "rh_palm",
    "palm",
    "rh_wrist",
    "wrist",
    "rh_forearm",
    "forearm",
]

FINGER_ACTUATORS = {
    "index": ["rh_A_FFJ3", "rh_A_FFJ0"],
    "middle": ["rh_A_MFJ3", "rh_A_MFJ0"],
    "ring": ["rh_A_RFJ3", "rh_A_RFJ0"],
    "pinky": ["rh_A_LFJ3", "rh_A_LFJ0"],
}

SIDE_ACTUATORS = {
    "index": "rh_A_FFJ4",
    "middle": "rh_A_MFJ4",
    "ring": "rh_A_RFJ4",
    "pinky": "rh_A_LFJ4",
}

SIDE_SIGN = {
    "index": 1.0,
    "middle": 1.0,
    "ring": -1.0,
    "pinky": -1.0,
}

SIDE_SCALE = {
    "index": 1.20,
    "middle": 1.00,
    "ring": 1.00,
    "pinky": 1.20,
}

THUMB_ACTUATORS = {
    "th5": "rh_A_THJ5",
    "th4": "rh_A_THJ4",
    "th3": "rh_A_THJ3",
    "th2": "rh_A_THJ2",
    "th1": "rh_A_THJ1",
}

WRIST_ACTUATORS = {
    "wrist_1": "rh_A_WRJ1",  # 앞뒤 굽힘, 지금은 고정
    "wrist_2": "rh_A_WRJ2",  # 좌우 흔들기
}

WRIST_LR_SCALE = 4.00
WRIST_LR_MAX_ANGLE_DEG = 25.0
WRIST_LR_DEADZONE = 0.03


@dataclass
class Landmark:
    x: float
    y: float
    z: float


# ============================================================
# 기본 계산 함수
# ============================================================

def angle_3points(a, b, c):
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-8:
        return 180.0

    cosang = np.dot(ba, bc) / denom
    cosang = np.clip(cosang, -1.0, 1.0)

    return math.degrees(math.acos(cosang))


def calc_bend(hand_landmarks, finger_name):
    ids = FINGER_LANDMARKS[finger_name]
    pts = []

    for idx in ids:
        lm = hand_landmarks[idx]
        pts.append([lm.x, lm.y, lm.z])

    if finger_name == "thumb":
        angle1 = angle_3points(pts[0], pts[1], pts[2])
        angle2 = angle_3points(pts[1], pts[2], pts[3])
        angle_avg = 0.5 * angle1 + 0.5 * angle2
        bend = (165.0 - angle_avg) / 65.0
        bend = bend * 3.0
    else:
        angle_pip = angle_3points(pts[0], pts[1], pts[2])
        angle_dip = angle_3points(pts[1], pts[2], pts[3])
        angle_avg = 0.6 * angle_pip + 0.4 * angle_dip
        bend = (170.0 - angle_avg) / 90.0

    bend = float(np.clip(bend, 0.0, 1.0))

    if finger_name == "index":
        bend = bend * 2.8
        bend = float(np.clip(bend, 0.0, 1.0))

    return bend


def calc_side(hand_landmarks, finger_name):
    if finger_name == "index":
        mcp_id = 5
        pip_id = 6
    elif finger_name == "middle":
        mcp_id = 9
        pip_id = 10
    elif finger_name == "ring":
        mcp_id = 13
        pip_id = 14
    elif finger_name == "pinky":
        mcp_id = 17
        pip_id = 18
    else:
        return 0.0

    mcp = hand_landmarks[mcp_id]
    pip = hand_landmarks[pip_id]

    index_mcp = hand_landmarks[5]
    pinky_mcp = hand_landmarks[17]

    palm_width = abs(index_mcp.x - pinky_mcp.x)
    if palm_width < 1e-5:
        palm_width = 0.1

    side = (pip.x - mcp.x) / palm_width

    deadzone = 0.02
    if abs(side) < deadzone:
        side = 0.0

    side = side * 2.0
    return float(np.clip(side, -1.0, 1.0))


def calc_wrist_lr(hand_landmarks):
    wrist = hand_landmarks[0]
    middle_mcp = hand_landmarks[9]

    vx = middle_mcp.x - wrist.x
    vy = middle_mcp.y - wrist.y

    lr_angle_deg = math.degrees(math.atan2(vx, -vy))
    wrist_lr = lr_angle_deg / WRIST_LR_MAX_ANGLE_DEG

    return float(np.clip(wrist_lr, -1.0, 1.0))


def landmark_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def calc_human_fingertips_norm(hand_landmarks):
    # 사람 손은 MediaPipe 손목 기준 상대좌표 + 손 크기 정규화
    wrist = landmark_to_np(hand_landmarks[0])
    index_mcp = landmark_to_np(hand_landmarks[5])
    pinky_mcp = landmark_to_np(hand_landmarks[17])

    human_scale = np.linalg.norm(index_mcp - pinky_mcp)
    if human_scale < 1e-8:
        human_scale = 1.0

    human_tip_norm = {}
    for finger in FINGERS:
        tip_id = FINGERTIP_LANDMARKS[finger]
        tip = landmark_to_np(hand_landmarks[tip_id])
        human_tip_norm[finger] = (tip - wrist) / human_scale

    return human_tip_norm


# ============================================================
# MuJoCo 관련 함수
# ============================================================

def actuator_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def set_actuator_fraction(model, data, name, fraction):
    aid = actuator_id(model, name)
    if aid < 0:
        return

    lo, hi = model.actuator_ctrlrange[aid]
    fraction = float(np.clip(fraction, 0.0, 1.0))
    data.ctrl[aid] = lo + fraction * (hi - lo)


def set_actuator_value(model, data, name, value):
    aid = actuator_id(model, name)
    if aid < 0:
        return

    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(value, lo, hi))


def find_mujoco_obj(model, candidates):
    for name in candidates:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id >= 0:
            return ("site", site_id, name)

    for name in candidates:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return ("body", body_id, name)

    return None


def get_mujoco_obj_pos(data, obj_info):
    obj_type, obj_id, _ = obj_info

    if obj_type == "site":
        return data.site_xpos[obj_id].copy()

    if obj_type == "body":
        return data.xpos[obj_id].copy()

    return None


def resolve_robot_fingertip_objects(model):
    robot_tip_objs = {}

    print("\nRobot fingertip object check")

    for finger in FINGERS:
        obj_info = find_mujoco_obj(model, ROBOT_TIP_CANDIDATES[finger])
        robot_tip_objs[finger] = obj_info

        if obj_info is None:
            print(f"{finger}: not found")
        else:
            obj_type, obj_id, obj_name = obj_info
            print(f"{finger}: {obj_type} '{obj_name}' id = {obj_id}")

    palm_obj = find_mujoco_obj(model, ROBOT_PALM_CANDIDATES)

    if palm_obj is None:
        print("palm: not found -> fingertip mean position will be used as palm reference")
    else:
        obj_type, obj_id, obj_name = palm_obj
        print(f"palm: {obj_type} '{obj_name}' id = {obj_id}")

    return robot_tip_objs, palm_obj


def calc_robot_fingertips_norm(data, robot_tip_objs, palm_obj, robot_scale):
    robot_tip_pos = {}

    for finger in FINGERS:
        obj_info = robot_tip_objs.get(finger)
        if obj_info is None:
            return None, robot_scale

        pos = get_mujoco_obj_pos(data, obj_info)
        if pos is None:
            return None, robot_scale

        robot_tip_pos[finger] = pos

    if palm_obj is not None:
        palm_pos = get_mujoco_obj_pos(data, palm_obj)
    else:
        palm_pos = np.mean(
            np.vstack([robot_tip_pos[finger] for finger in FINGERS]),
            axis=0,
        )

    if palm_pos is None:
        return None, robot_scale

    if robot_scale is None:
        scale = np.linalg.norm(robot_tip_pos["index"] - robot_tip_pos["pinky"])
        if scale < 1e-8:
            scale = 1.0
        robot_scale = scale

    robot_tip_norm = {}
    for finger in FINGERS:
        robot_tip_norm[finger] = (robot_tip_pos[finger] - palm_pos) / robot_scale

    return robot_tip_norm, robot_scale


def apply_shadow_hand_control(model, data, bends, sides, wrist_lr_smooth):
    for finger, acts in FINGER_ACTUATORS.items():
        bend = bends[finger]

        mcp_act = acts[0]
        pip_dip_act = acts[1]

        if finger == "index":
            set_actuator_fraction(model, data, mcp_act, 1.20 * bend)
            set_actuator_fraction(model, data, pip_dip_act, 1.50 * bend)
        else:
            set_actuator_fraction(model, data, mcp_act, 0.75 * bend)
            set_actuator_fraction(model, data, pip_dip_act, bend)

    for finger, side_act in SIDE_ACTUATORS.items():
        side_value = sides[finger]
        side_value = side_value * SIDE_SIGN[finger] * SIDE_SCALE[finger]
        set_actuator_value(model, data, side_act, side_value)

    thumb = bends["thumb"]
    thumb = float(np.clip(thumb * 1.8, 0.0, 1.0))

    set_actuator_value(model, data, THUMB_ACTUATORS["th5"], 0.65)
    set_actuator_fraction(model, data, THUMB_ACTUATORS["th4"], 0.20 + 0.80 * thumb)
    set_actuator_fraction(model, data, THUMB_ACTUATORS["th3"], 0.50 + 0.45 * thumb)
    set_actuator_fraction(model, data, THUMB_ACTUATORS["th2"], 0.15 + 0.85 * thumb)
    set_actuator_fraction(model, data, THUMB_ACTUATORS["th1"], 0.10 + 0.90 * thumb)

    # 손목 앞뒤는 고정, 좌우만 적용
    set_actuator_value(model, data, WRIST_ACTUATORS["wrist_1"], 0.0)
    set_actuator_value(model, data, WRIST_ACTUATORS["wrist_2"], -WRIST_LR_SCALE * wrist_lr_smooth)


# ============================================================
# CSV / 오차 계산
# ============================================================

def make_fingertip_csv_header():
    header = [
        "time_s",
        "alpha",
        "thumb_raw",
        "index_raw",
        "middle_raw",
        "ring_raw",
        "pinky_raw",
    ]

    for prefix in ["human", "robot"]:
        for finger in FINGERS:
            header += [
                f"{prefix}_{finger}_x",
                f"{prefix}_{finger}_y",
                f"{prefix}_{finger}_z",
            ]

    for finger in FINGERS:
        header.append(f"{finger}_tip_error")

    header.append("overall_tip_error")

    return header


def make_fingertip_csv_row(t, alpha, raw_bends, human_tip_norm, robot_tip_norm, tip_errors, overall_error):
    row = [
        t,
        alpha,
        raw_bends["thumb"],
        raw_bends["index"],
        raw_bends["middle"],
        raw_bends["ring"],
        raw_bends["pinky"],
    ]

    for source in [human_tip_norm, robot_tip_norm]:
        for finger in FINGERS:
            row += [
                source[finger][0],
                source[finger][1],
                source[finger][2],
            ]

    for finger in FINGERS:
        row.append(tip_errors[finger])

    row.append(overall_error)

    return row


def read_landmark_from_row(row, idx):
    x_col = f"lm{idx}_x"
    y_col = f"lm{idx}_y"
    z_col = f"lm{idx}_z"

    if x_col not in row.index or y_col not in row.index or z_col not in row.index:
        raise KeyError(
            f"raw_hand_landmarks.csv에서 {x_col}, {y_col}, {z_col} 컬럼을 찾지 못했습니다."
        )

    return Landmark(float(row[x_col]), float(row[y_col]), float(row[z_col]))


def row_to_hand_landmarks(row):
    return [read_landmark_from_row(row, i) for i in range(21)]


def calc_fingertip_errors(human_tip_norm, robot_tip_norm, human_ref, robot_ref):
    errors = {}

    for finger in FINGERS:
        # 사람/로봇의 절대 크기와 초기 위치 차이를 줄이기 위한 방식
        human_move = human_tip_norm[finger] - human_ref[finger]
        robot_move = robot_tip_norm[finger] - robot_ref[finger]
        errors[finger] = float(np.linalg.norm(human_move - robot_move))

    overall_error = float(np.mean([errors[finger] for finger in EVAL_FINGERS]))

    return errors, overall_error


def rmse(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(values ** 2)))


def alpha_to_tag(alpha):
    # 0.1 -> 01, 0.9 -> 09, 1.0 -> 10
    return f"{int(round(alpha * 10)):02d}"


# ============================================================
# alpha replay
# ============================================================

def replay_for_alpha(raw_df, model, robot_tip_objs, robot_palm_obj, alpha):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bends = {finger: 0.0 for finger in FINGERS}
    raw_bends = {finger: 0.0 for finger in FINGERS}
    sides = {"index": 0.0, "middle": 0.0, "ring": 0.0, "pinky": 0.0}

    wrist_lr_smooth = 0.0
    wrist_lr_zero = None

    robot_scale = None
    human_tip_ref = None
    robot_tip_ref = None

    output_rows = []
    start_t = float(raw_df["time_s"].iloc[0])

    # raw CSV의 실제 시간 간격과 MuJoCo timestep을 동기화
    # 실시간 코드와 같은 기준으로, 실제 지난 시간만큼 mj_step을 누적 실행
    prev_t = 0.0
    sim_time_buffer = 0.0
    mj_dt = float(model.opt.timestep)

    for _, row in raw_df.iterrows():
        t = float(row["time_s"]) - start_t
        dt_raw = max(0.0, t - prev_t)
        hand_landmarks = row_to_hand_landmarks(row)

        for finger in bends.keys():
            raw_bend = calc_bend(hand_landmarks, finger)
            raw_bends[finger] = raw_bend
            bends[finger] = (1.0 - alpha) * bends[finger] + alpha * raw_bend

        for finger in sides.keys():
            raw_side = calc_side(hand_landmarks, finger)
            sides[finger] = (1.0 - alpha) * sides[finger] + alpha * raw_side

        raw_wrist_lr = calc_wrist_lr(hand_landmarks)

        if wrist_lr_zero is None:
            wrist_lr_zero = raw_wrist_lr

        raw_wrist_lr = raw_wrist_lr - wrist_lr_zero

        if abs(raw_wrist_lr) < WRIST_LR_DEADZONE:
            raw_wrist_lr = 0.0

        raw_wrist_lr = float(np.clip(raw_wrist_lr * 2.8, -1.0, 1.0))
        wrist_lr_smooth = (1.0 - alpha) * wrist_lr_smooth + alpha * raw_wrist_lr

        human_tip_norm = calc_human_fingertips_norm(hand_landmarks)

        apply_shadow_hand_control(model, data, bends, sides, wrist_lr_smooth)

        sim_time_buffer += dt_raw

        while sim_time_buffer >= mj_dt:
            mujoco.mj_step(model, data)
            sim_time_buffer -= mj_dt

        prev_t = t

        robot_tip_norm, robot_scale = calc_robot_fingertips_norm(
            data,
            robot_tip_objs,
            robot_palm_obj,
            robot_scale,
        )

        if robot_tip_norm is None:
            raise RuntimeError("로봇 손끝 site/body를 찾지 못해서 replay 분석을 진행할 수 없습니다.")

        if human_tip_ref is None:
            human_tip_ref = {finger: human_tip_norm[finger].copy() for finger in FINGERS}

        if robot_tip_ref is None:
            robot_tip_ref = {finger: robot_tip_norm[finger].copy() for finger in FINGERS}

        tip_errors, overall_tip_error = calc_fingertip_errors(
            human_tip_norm,
            robot_tip_norm,
            human_tip_ref,
            robot_tip_ref,
        )

        output_rows.append(
            make_fingertip_csv_row(
                t,
                alpha,
                raw_bends,
                human_tip_norm,
                robot_tip_norm,
                tip_errors,
                overall_tip_error,
            )
        )

    result_df = pd.DataFrame(output_rows, columns=make_fingertip_csv_header())
    return result_df


def summarize_alpha_result(alpha, result_df):
    eval_df = result_df[result_df["time_s"] >= EVAL_START_TIME].copy()

    if eval_df.empty:
        eval_df = result_df.copy()

    error_cols = [f"{finger}_tip_error" for finger in EVAL_FINGERS]
    errors = eval_df[error_cols].to_numpy(dtype=float)

    row = {
        "alpha": alpha,
        "rmse_all": rmse(errors.reshape(-1)),
        "mean_error_all": float(np.mean(errors)),
        "max_error_all": float(np.max(errors)),
    }

    for finger in EVAL_FINGERS:
        e = eval_df[f"{finger}_tip_error"].to_numpy(dtype=float)
        row[f"rmse_{finger}"] = rmse(e)
        row[f"mean_{finger}"] = float(np.mean(e))
        row[f"max_{finger}"] = float(np.max(e))

    return row


# ============================================================
# main
# ============================================================

def main():
    raw_df = pd.read_csv(RAW_LANDMARK_CSV)
    raw_df = raw_df.dropna().copy()

    if raw_df.empty:
        raise RuntimeError("raw_hand_landmarks.csv가 비어 있습니다. 측정 코드를 다시 실행해서 손을 충분히 움직여 주세요.")

    print("raw_hand_landmarks:", raw_df.shape)

    model = mujoco.MjModel.from_xml_path(MUJOCO_MODEL_PATH)
    robot_tip_objs, robot_palm_obj = resolve_robot_fingertip_objects(model)

    summary_rows = []

    for alpha in ALPHAS:
        print(f"\nReplay alpha = {alpha}")

        result_df = replay_for_alpha(
            raw_df,
            model,
            robot_tip_objs,
            robot_palm_obj,
            alpha,
        )

        out_csv = f"fingertip_position_log_a{alpha_to_tag(alpha)}.csv"
        result_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print("saved:", out_csv)

        summary_rows.append(summarize_alpha_result(alpha, result_df))

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("alpha").reset_index(drop=True)

    best_alpha = float(summary.loc[summary["rmse_all"].idxmin(), "alpha"])
    summary["is_best_rmse"] = summary["alpha"] == best_alpha

    print("\nAlpha RMSE summary")
    print(summary)
    print("\nBest alpha by RMSE =", best_alpha)

    summary.to_csv("alpha_rmse_summary.csv", index=False, encoding="utf-8-sig")
    print("saved: alpha_rmse_summary.csv")

    print("\n다음 명령으로 그래프를 생성하세요:")
    print("python 03_plot.py")


if __name__ == "__main__":
    main()
