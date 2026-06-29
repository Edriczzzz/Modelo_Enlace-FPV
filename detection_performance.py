"""
detection_performance.py
========================
Script de detección YOLO + métricas de calidad de video.
Adaptado del script original del Dr. Vera-Amaro.

USO:
    python detection_performance.py --video videos/analogico/video_1.mp4 --id video_analog_1 --tipo analogico
    python detection_performance.py --video videos/digital/video_d1.mp4  --id video_digital_1 --tipo digital --show

REQUISITOS:
    pip install ultralytics opencv-python numpy pandas matplotlib
"""

import os, time, argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO


# ============================================================
# CONFIGURACIÓN
# ============================================================
MODEL_PATH           = "yolov8n.pt"
TARGET_CLASSES       = ["person"]
CONF_THRESHOLD       = 0.35
LATENCY_THRESHOLD_MS = 200
WINDOW_SECONDS       = 1.0
WARMUP_FRAMES        = 10
MAX_FRAMES           = None
SAVE_ANNOTATED_VIDEO = False


# ============================================================
# MÉTRICAS VISUALES  (igual al script del profe)
# ============================================================
def compute_gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def compute_brightness(gray):
    return float(np.mean(gray))

def compute_contrast(gray):
    return float(np.std(gray))

def compute_blur(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def compute_noise(gray):
    median = cv2.medianBlur(gray, 3)
    return float(np.std(gray.astype(np.float32) - median.astype(np.float32)))

def compute_temporal_instability(prev_gray, gray):
    if prev_gray is None:
        return 0.0
    return float(np.mean(cv2.absdiff(prev_gray, gray)))

def normalize(value, min_val, max_val):
    return (max(min_val, min(value, max_val)) - min_val) / (max_val - min_val)

def compute_video_degradation(blur, contrast, brightness, noise, temporal_instability):
    blur_deg       = 1.0 - normalize(blur, 30, 180)
    contrast_deg   = 1.0 - normalize(contrast, 20, 90)
    brightness_deg = 1.0 - (1.0 - abs(brightness - 127.5) / 127.5)
    noise_deg      = normalize(noise, 0, 30)
    temporal_deg   = normalize(temporal_instability, 0, 80)
    d = 0.30*blur_deg + 0.20*contrast_deg + 0.15*brightness_deg + 0.20*noise_deg + 0.15*temporal_deg
    return float(max(0.0, min(d, 1.0)))


# ============================================================
# VENTANAS  (igual al script del profe)
# ============================================================
def compute_window_metrics(df, video_fps):
    ws = max(1, int(WINDOW_SECONDS * video_fps))
    df = df.copy()
    df["window_id"] = (df["frame_id"] // ws).astype(int)
    window_df = df.groupby("window_id").agg(
        video_id                  =("video_id","first"),
        tipo                      =("tipo","first"),
        start_frame               =("frame_id","min"),
        end_frame                 =("frame_id","max"),
        start_time_s              =("time_s","min"),
        end_time_s                =("time_s","max"),
        frames                    =("frame_id","count"),
        D_video_mean              =("D_video_obs","mean"),
        D_video_max               =("D_video_obs","max"),
        R_det_exp                 =("detection_success","mean"),
        R_det_conf                =("R_det_frame","mean"),
        max_confidence_mean       =("max_confidence","mean"),
        blur_mean                 =("blur","mean"),
        noise_mean                =("noise","mean"),
        temporal_instability_mean =("temporal_instability","mean"),
        contrast_mean             =("contrast","mean"),
        brightness_mean           =("brightness","mean"),
        latency_mean_ms           =("total_latency_ms","mean"),
        latency_p95_ms            =("total_latency_ms", lambda x: x.quantile(0.95)),
        R_latency                 =("latency_success","mean"),
        R_system_proxy            =("R_system_proxy","mean"),
    ).reset_index()
    return window_df


# ============================================================
# RESUMEN
# ============================================================
def compute_summary(df, window_df, video_id, tipo, video_fps, width, height, runtime_s):
    n = len(df)
    return pd.DataFrame([{
        "video_id"               : video_id,
        "tipo"                   : tipo,
        "resolution"             : f"{width}x{height}",
        "video_fps_original"     : video_fps,
        "processed_frames"       : n,
        "real_processing_fps"    : n / runtime_s if runtime_s > 0 else 0,
        "R_detection_frame_mean" : df["detection_success"].mean(),
        "R_detection_conf_mean"  : df["R_det_frame"].mean(),
        "R_detection_window_mean": window_df["R_det_exp"].mean(),
        "D_video_mean"           : df["D_video_obs"].mean(),
        "D_video_std"            : df["D_video_obs"].std(),
        "D_video_min"            : df["D_video_obs"].min(),
        "D_video_max"            : df["D_video_obs"].max(),
        "R_latency_200ms"        : (df["total_latency_ms"] <= 200).mean(),
        "latency_mean_ms"        : df["total_latency_ms"].mean(),
        "latency_median_ms"      : df["total_latency_ms"].median(),
        "latency_p95_ms"         : df["total_latency_ms"].quantile(0.95),
        "max_confidence_mean"    : df["max_confidence"].mean(),
        "R_system_proxy_mean"    : df["R_system_proxy"].mean(),
        "blur_mean"              : df["blur"].mean(),
        "noise_mean"             : df["noise"].mean(),
        "temporal_instability_mean": df["temporal_instability"].mean(),
        "contrast_mean"          : df["contrast"].mean(),
        "brightness_mean"        : df["brightness"].mean(),
    }])


# ============================================================
# GRÁFICAS POR VIDEO
# ============================================================
def generate_plots(df_eval, window_df, plots_dir, video_id, tipo):
    col = "#1f77b4" if "analog" in tipo.lower() else "#2ca02c"

    def save(fig, name):
        fig.savefig(os.path.join(plots_dir, name), dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(window_df["start_time_s"], window_df["D_video_mean"], "o-", color=col, label=r"$D_{video}^{obs}$")
    ax.plot(window_df["start_time_s"], window_df["R_det_exp"], "s--", color="orange", label=r"$R_{det}^{exp}$")
    ax.set_xlabel("Time [s]"); ax.set_ylabel("Value")
    ax.set_title(f"Video Degradation & Detection Reliability — {video_id} ({tipo})")
    ax.grid(True); ax.legend(); fig.tight_layout(); save(fig,"01_Dvideo_Rdet_time.png")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(window_df["D_video_mean"], window_df["R_det_exp"], alpha=0.75, color=col)
    ax.set_xlabel(r"$D_{video}^{obs}$"); ax.set_ylabel(r"$R_{det}^{exp}$")
    ax.set_title(f"Detection Reliability vs Video Degradation — {video_id}")
    ax.grid(True); fig.tight_layout(); save(fig,"02_Rdet_vs_Dvideo.png")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(window_df["D_video_mean"], window_df["max_confidence_mean"], alpha=0.75, color=col)
    ax.set_xlabel(r"$D_{video}^{obs}$"); ax.set_ylabel("Mean YOLO Confidence")
    ax.set_title(f"YOLO Confidence vs Video Degradation — {video_id}")
    ax.grid(True); fig.tight_layout(); save(fig,"03_confidence_vs_Dvideo.png")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(window_df["start_time_s"], window_df["R_system_proxy"], "o-", color=col)
    ax.set_xlabel("Time [s]"); ax.set_ylabel("System Reliability Proxy")
    ax.set_title(f"System Reliability Proxy — {video_id} ({tipo})")
    ax.grid(True); fig.tight_layout(); save(fig,"04_Rsystem_time.png")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.hist(df_eval["total_latency_ms"], bins=40, color=col)
    ax.set_xlabel("Latency [ms]"); ax.set_ylabel("Frequency")
    ax.set_title(f"Processing Latency Distribution — {video_id}")
    ax.grid(True); fig.tight_layout(); save(fig,"05_latency_distribution.png")

    thresholds  = np.arange(50, 401, 25)
    r_lat_vals  = [(df_eval["total_latency_ms"] <= th).mean() for th in thresholds]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(thresholds, r_lat_vals, "o-", color=col)
    ax.set_xlabel("Latency Threshold [ms]"); ax.set_ylabel(r"$R_{lat}$")
    ax.set_title(f"Latency Reliability vs Threshold — {video_id}")
    ax.grid(True); fig.tight_layout(); save(fig,"06_Rlatency_threshold.png")


# ============================================================
# ANÁLISIS PRINCIPAL
# ============================================================
def analyze_video(video_path, video_id, tipo, show_video=False):
    output_dir = os.path.join("resultados_modelo_fpv", video_id)
    plots_dir  = os.path.join(output_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(plots_dir,  exist_ok=True)

    model       = YOLO(MODEL_PATH)
    class_names = model.names
    target_ids  = [cid for cid,name in class_names.items() if name in TARGET_CLASSES]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_fr  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*55}")
    print(f"  Video : {video_path}")
    print(f"  ID    : {video_id}  |  Tipo: {tipo}")
    print(f"  FPS   : {video_fps:.1f}  |  Frames: {total_fr}  |  {width}x{height}")
    print(f"{'='*55}")

    out_video = None
    if SAVE_ANNOTATED_VIDEO:
        out_video = cv2.VideoWriter(
            os.path.join(output_dir, f"{video_id}_annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (width, height))

    frame_rows = []
    prev_gray  = None
    frame_id   = 0
    t_global   = time.perf_counter()

    while True:
        if MAX_FRAMES and frame_id >= MAX_FRAMES:
            break
        t0 = time.perf_counter()
        ret, frame = cap.read()
        t1 = time.perf_counter()
        if not ret:
            break

        gray                 = compute_gray(frame)
        brightness           = compute_brightness(gray)
        contrast             = compute_contrast(gray)
        blur                 = compute_blur(gray)
        noise                = compute_noise(gray)
        temporal_instability = compute_temporal_instability(prev_gray, gray)
        d_video_obs          = compute_video_degradation(blur, contrast, brightness, noise, temporal_instability)

        t2           = time.perf_counter()
        yolo_results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        t3           = time.perf_counter()

        result      = yolo_results[0]
        boxes       = result.boxes
        confidences = []
        det_classes = []
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf   = float(box.conf[0])
                if cls_id in target_ids:
                    confidences.append(conf)
                    det_classes.append(class_names[cls_id])

        num_det          = len(confidences)
        detection_success= 1 if num_det > 0 else 0
        max_confidence   = float(np.max(confidences)) if confidences else 0.0
        avg_confidence   = float(np.mean(confidences)) if confidences else 0.0
        read_lat_ms      = (t1-t0)*1000
        inf_lat_ms       = (t3-t2)*1000
        total_lat_ms     = (t3-t0)*1000
        lat_success      = 1 if total_lat_ms <= LATENCY_THRESHOLD_MS else 0
        r_det_frame      = detection_success * max_confidence
        r_system_proxy   = (1.0 - d_video_obs) * r_det_frame * lat_success

        frame_rows.append({
            "video_id"              : video_id,
            "tipo"                  : tipo,
            "frame_id"              : frame_id,
            "time_s"                : frame_id / video_fps,
            "brightness"            : brightness,
            "contrast"              : contrast,
            "blur"                  : blur,
            "noise"                 : noise,
            "temporal_instability"  : temporal_instability,
            "D_video_obs"           : d_video_obs,
            "num_detections"        : num_det,
            "detection_success"     : detection_success,
            "max_confidence"        : max_confidence,
            "avg_confidence"        : avg_confidence,
            "R_det_frame"           : r_det_frame,
            "read_latency_ms"       : read_lat_ms,
            "inference_latency_ms"  : inf_lat_ms,
            "total_latency_ms"      : total_lat_ms,
            "latency_success"       : lat_success,
            "R_system_proxy"        : r_system_proxy,
            "detected_classes"      : ",".join(det_classes),
        })

        if show_video or out_video:
            annotated = result.plot()
            cv2.putText(annotated,
                f"Dv={d_video_obs:.2f} | Rdet={r_det_frame:.2f} | Lat={total_lat_ms:.1f}ms | {tipo}",
                (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            if out_video:
                out_video.write(annotated)
            if show_video:
                cv2.imshow(f"FPV-YOLO | {video_id}", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        prev_gray = gray.copy()
        frame_id += 1

    runtime = time.perf_counter() - t_global
    cap.release()
    if out_video:
        out_video.release()
    cv2.destroyAllWindows()

    df      = pd.DataFrame(frame_rows)
    df_eval = df[df["frame_id"] >= WARMUP_FRAMES].copy()

    df_eval.to_csv(os.path.join(output_dir, "frames_metrics.csv"), index=False)
    window_df = compute_window_metrics(df_eval, video_fps)
    window_df.to_csv(os.path.join(output_dir, "window_metrics.csv"), index=False)
    summary = compute_summary(df_eval, window_df, video_id, tipo, video_fps, width, height, runtime)
    summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)
    generate_plots(df_eval, window_df, plots_dir, video_id, tipo)

    print(f"  ✅ {frame_id} frames en {runtime:.1f}s")
    print(f"     D_video : {df_eval['D_video_obs'].mean():.4f}")
    print(f"     C_YOLO  : {df_eval['max_confidence'].mean():.4f}")
    print(f"     R_sys   : {df_eval['R_system_proxy'].mean():.4f}")
    print(f"     Salida  : {output_dir}/\n")
    return window_df


# ============================================================
# PUNTO DE ENTRADA
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detección FPV + métricas (estilo Dr. Vera-Amaro)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--id",    required=True,  help="ID único del video, ej: video_analog_1")
    parser.add_argument("--tipo",  default="analogico", choices=["analogico","digital"])
    parser.add_argument("--show",  action="store_true", help="Mostrar video en pantalla")
    args = parser.parse_args()
    analyze_video(args.video, args.id, args.tipo, args.show)