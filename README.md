# Webcam-based Shadow Hand Avatar Simulation

## Project Overview

This project implements a webcam-based Shadow Hand avatar simulation using MediaPipe and MuJoCo. Human hand motion is captured from a webcam, converted into finger bending, side-spreading, and wrist lateral motion values, and mapped to the actuator inputs of the MuJoCo Shadow Hand model.

The objective of this project is to retarget human hand motion to a robot hand with different degrees of freedom and evaluate its tracking performance using fingertip-based metrics.

---

## System Pipeline

```text
Webcam Input
      ↓
MediaPipe Hand Landmark Detection
      ↓
21 Hand Landmark Positions
      ↓
Landmark-based Motion Value Calculation
      ↓
Low-Pass Filter
      ↓
Shadow Hand Actuator Mapping
      ↓
MuJoCo Simulation
      ↓
Fingertip Tracking Evaluation
```

---

## Tech Stack

| Category | Tool |
|---|---|
| Development Environment | VS Code |
| Language | Python |
| Webcam Processing | OpenCV |
| Hand Landmark Detection | MediaPipe Hand Landmarker |
| Robot Hand Model | Shadow Hand robot E3M5 |
| Simulation Environment | MuJoCo |
| Data Analysis | NumPy, Pandas, Matplotlib |

---

## Key Implementation

### Variable Definition

| Symbol | Code Variable | Description |
|---|---|---|
| \( P_i \) | `hand_landmarks[i]` | i-th MediaPipe hand landmark |
| \( P_i=(x_i,y_i,z_i) \) | `lm.x, lm.y, lm.z` | 3D coordinate of landmark \(P_i\) |
| \( \mathbf{u}, \mathbf{v} \) | `ba`, `bc` | Adjacent finger bone vectors |
| \( \theta \) | `angle` | Joint angle between two vectors |
| \( b_f \) | `bend` | Normalized bending value of finger \(f\) |
| \( s_f \) | `side` | Normalized side-spreading value of finger \(f\) |
| \( W_p \) | `palm_width` | Palm width used for normalization |
| \( \mathbf{v}_{palm} \) | `vpalm` | Direction vector from wrist to middle MCP |
| \( w_{lr} \) | `wrist_lr` | Wrist left-right motion value |
| \( \alpha \) | `alpha` | Smoothing parameter of the low-pass filter |
| \( u_t^{raw} \) | `raw_bend`, `raw_side`, `raw_wrist_lr` | Raw motion value at frame \(t\) |
| \( u_t \) | `filtered_value` | Filtered motion value at frame \(t\) |
| \( c_i \) | `data.ctrl[i]` | MuJoCo actuator command |
| \( c_{min}, c_{max} \) | `lo`, `hi` | Minimum and maximum actuator control values |
| \( e_f \) | `tip_errors[finger]` | Fingertip tracking error of finger \(f\) |
| \( RMSE \) | `rmse_all` | Root mean square error of fingertip tracking |
| \( Delay \) | `delay_s` | Response delay between human and robot motion |
| \( Score \) | `score` | Final smoothing parameter selection score |

---

## Hand Landmark Extraction

Webcam frames were captured using OpenCV and processed with MediaPipe Hand Landmarker. MediaPipe extracts 21 hand landmark positions, including the wrist and finger joints.

Each landmark is defined as:

\[
P_i = (x_i, y_i, z_i), \quad i=0,1,\dots,20
\]

where \(P_0\) is the wrist landmark and the remaining landmarks represent the joints of the thumb, index, middle, ring, and pinky fingers.

The extracted landmarks were used to calculate finger bending, finger side-spreading, and wrist lateral motion values.

MediaPipe uses an image-based coordinate system. The \(x\)-axis represents the horizontal image direction, the \(y\)-axis represents the vertical image direction, and the \(z\)-axis represents relative depth.

---

## Landmark-based Motion Value Calculation

### Finger Bending Value

Finger bending was calculated using the angle between two adjacent finger bone vectors.

For three adjacent landmarks \(P_a\), \(P_b\), and \(P_c\), two vectors are defined as:

\[
\mathbf{u} = P_a - P_b
\]

\[
\mathbf{v} = P_c - P_b
\]

The joint angle is calculated by:

\[
\theta =
\cos^{-1}
\left(
\frac{\mathbf{u} \cdot \mathbf{v}}
{\|\mathbf{u}\|\|\mathbf{v}\|}
\right)
\]

For the index, middle, ring, and pinky fingers, the average bending angle was calculated using the PIP and DIP joint angles:

\[
\theta_f =
0.6\theta_{PIP}
+
0.4\theta_{DIP}
\]

The bending value was then normalized as:

\[
b_f =
\text{clip}
\left(
\frac{170^\circ - \theta_f}
{90^\circ},
0,
1
\right)
\]

where \(b_f\) represents the bending value of finger \(f\). A value close to 0 indicates an extended finger, while a value close to 1 indicates a bent finger.

For the thumb, two thumb joint angles were averaged separately because the thumb has a different joint structure from the other fingers:

\[
\theta_{thumb} =
0.5\theta_{MCP}
+
0.5\theta_{IP}
\]

\[
b_{thumb} =
\text{clip}
\left(
3.0 \times
\frac{165^\circ - \theta_{thumb}}
{65^\circ},
0,
1
\right)
\]

---

### Finger Side-spreading Value

Finger side-spreading was estimated using the lateral displacement between the MCP and PIP landmarks.

The palm width was defined as:

\[
W_p = |x_5 - x_{17}|
\]

where \(x_5\) is the x-coordinate of the index MCP landmark and \(x_{17}\) is the x-coordinate of the pinky MCP landmark.

For each finger, the side-spreading value was calculated as:

\[
s_f =
\frac{x_{PIP} - x_{MCP}}
{W_p}
\]

The side-spreading values for each finger are:

\[
s_{index} =
\frac{x_6 - x_5}
{W_p}
\]

\[
s_{middle} =
\frac{x_{10} - x_9}
{W_p}
\]

\[
s_{ring} =
\frac{x_{14} - x_{13}}
{W_p}
\]

\[
s_{pinky} =
\frac{x_{18} - x_{17}}
{W_p}
\]

In the implementation, a dead zone and scaling factor were applied to reduce small jitter and adjust the side motion sensitivity:

\[
s_f =
\text{clip}
\left(
k_s s_f,
-1,
1
\right)
\]

Here, \(s_f\) is not a direct joint angle. It is a normalized lateral spreading value calculated from the landmark coordinates.

---

### Wrist Left-right Motion Value

Wrist left-right motion was estimated using the direction vector from the wrist landmark \(P_0\) to the middle finger MCP landmark \(P_9\).

\[
\mathbf{v}_{palm} = P_9 - P_0
\]

\[
v_x = x_9 - x_0
\]

\[
v_y = y_9 - y_0
\]

The wrist lateral angle was calculated using:

\[
\theta_w =
\text{atan2}
\left(
v_x,
-v_y
\right)
\]

The wrist left-right value was then normalized by the maximum wrist angle:

\[
w_{lr} =
\text{clip}
\left(
\frac{\theta_w}
{\theta_{w,max}},
-1,
1
\right)
\]

The initial wrist direction was used as a neutral reference, so the final wrist motion value represents the relative change from the initial hand posture.

---

## Low-Pass Filter

MediaPipe landmark data can contain frame-to-frame noise. To reduce jitter, a first-order low-pass filter was applied to the calculated motion values.

\[
u_t =
(1-\alpha)u_{t-1}
+
\alpha u_t^{raw}
\]

where \(u_t^{raw}\) is the raw motion value calculated from the current frame, \(u_t\) is the filtered value, and \(\alpha\) is the smoothing parameter.

In this project, \(u\) can represent:

\[
u \in \{b_f, s_f, w_{lr}\}
\]

A smaller smoothing parameter produces smoother motion but increases response delay. A larger smoothing parameter improves responsiveness but becomes more sensitive to landmark noise.

Therefore, different smoothing parameter values were tested to select an appropriate value for real-time hand avatar control.

---

## MuJoCo Actuator Mapping

The proposed method does not directly control fingertip positions. Instead, the calculated motion values are mapped to Shadow Hand actuator commands.

MuJoCo controls the robot hand through actuator input values stored in `data.ctrl`.

For actuators controlled by a normalized fraction, the command was calculated as:

\[
c_i =
c_{min,i}
+
r_i
\left(
c_{max,i} - c_{min,i}
\right)
\]

where \(c_i\) is the actuator command, \(c_{min,i}\) and \(c_{max,i}\) are the minimum and maximum actuator control values, and \(r_i\) is the normalized input ratio.

For side and wrist actuators, the filtered values were directly scaled and clipped within the actuator control range:

\[
c_i =
\text{clip}
\left(
k_i u_t,
c_{min,i},
c_{max,i}
\right)
\]

The final actuator command was applied to MuJoCo using:

\[
data.ctrl[i] = c_i
\]

Through this mapping, human hand motion calculated from MediaPipe landmarks was converted into Shadow Hand actuator commands.

---

## Simulation Result & Analysis

### 1. Input Trajectory

The input trajectory was used to verify whether the human hand motion was correctly captured from the webcam.

Fingertip displacement was calculated from the initial position of each finger. During repeated grasping and opening motion, the displacement increased and decreased periodically.

This confirms that the MediaPipe landmark data properly captured the input hand motion.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_01_input_trajectory.png" width="700">
</p>

---

### 2. Tracking Comparison

The tracking comparison graph compares the fingertip displacement of the human hand and the MuJoCo Shadow Hand.

The robot hand displacement changed according to the grasping and opening motion of the human hand. This shows that the MediaPipe landmark-based motion values were successfully converted into Shadow Hand avatar motion.

Since the human hand and Shadow Hand have different joint structures and link lengths, the trajectories are not completely identical. The main evaluation focus was the overall tracking trend rather than perfect position matching.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_02_tracking_comparison.png" width="700">
</p>

---

### 3. Fingertip Error

Fingertip error was calculated to quantitatively evaluate the tracking performance of the robot hand.

For evaluation, both human and robot fingertip positions were normalized using their own reference frames and scales.

The fingertip error for finger \(f\) was calculated as:

\[
e_f(t) =
\left\|
\Delta P_f^{human}(t)
-
\Delta P_f^{robot}(t)
\right\|
\]

where:

\[
\Delta P_f(t) =
P_f(t) - P_f(0)
\]

The overall fingertip error was calculated as the average error of the index, middle, ring, and pinky fingers:

\[
e_{overall}(t) =
\frac{1}{4}
\sum_{f \in \{index,middle,ring,pinky\}}
e_f(t)
\]

The thumb was analyzed separately because thumb opposition and rotation are more complex than the other fingers.

The error increased during fast transition motions, such as grasping and opening, and decreased when the hand posture became stable.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_03_fingertip_error.png" width="700">
</p>

---

### 4. Smoothing Parameter Selection

The smoothing parameter was selected by comparing RMSE and response delay.

RMSE represents the overall fingertip tracking error, while delay represents the response lag between the human hand input and the robot hand motion.

The RMSE of fingertip tracking was calculated as:

\[
RMSE =
\sqrt{
\frac{1}{N}
\sum_{t=1}^{N}
e_{overall}(t)^2
}
\]

Because RMSE and delay have different units, min-max normalization was applied:

\[
\hat{x} =
\frac{x - x_{min}}
{x_{max} - x_{min}}
\]

The final score was calculated using equal weights for normalized RMSE and normalized delay:

\[
Score =
0.5\hat{RMSE}
+
0.5\hat{Delay}
\]

The smoothing parameter with the lowest score was selected.

A smaller smoothing parameter produced smoother motion but increased response delay. A larger smoothing parameter improved responsiveness but became more sensitive to landmark noise.

<p align="center">
  <img src="timesync_alpha_fingertip_analysis_figures/timesync_04_alpha_selection.png" width="700">
</p>

---

## Result Summary

This project implemented a webcam-based Shadow Hand avatar simulation using MediaPipe and MuJoCo.

Human hand landmarks were extracted from webcam input, and the landmark coordinates were converted into finger bending, side-spreading, and wrist lateral motion values. These values were smoothed using a low-pass filter and mapped to the actuator inputs of the MuJoCo Shadow Hand model.

The tracking performance was evaluated using fingertip error, RMSE, response delay, and smoothing parameter score. The final smoothing parameter was selected by considering both tracking accuracy and response delay.

---

## Limitations and Future Work

The human hand and Shadow Hand have different joint structures and degrees of freedom, which makes exact fingertip matching difficult.

Thumb motion is especially difficult to reproduce because it includes opposition and complex rotation. MediaPipe landmark noise can also affect the stability of the robot hand motion.

The current implementation uses an angle-based actuator mapping method instead of directly solving inverse kinematics for fingertip positions. This approach provides stable real-time avatar control, but it does not guarantee exact fingertip position matching.

Future work will focus on improving the motion retargeting accuracy using inverse kinematics, robot landmark correspondence, or learning-based mapping methods.
