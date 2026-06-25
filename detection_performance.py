"""
detection_performance.py
========================
Script 1: Detección YOLO + extracción de métricas de calidad de video
Basado en el modelo del paper:
  "A Communication-Perception Reliability Model for YOLO-Based Human Detection in UAV FPV Systems"
  R. Vera-Amaro, IPN UPIITA

USO:
    python detection_performance.py --video ruta/al/video.mp4 --output resultados.csv

REQUISITOS:
    pip install ultralytics opencv-python numpy pandas
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import time
import os
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# PARÁMETROS GLOBALES (ajusta si es necesario)
# ─────────────────────────────────────────────────────────────
CONF_THRESHOLD  = 0.25      # umbral mínimo de confianza YOLO
IOU_THRESHOLD   = 0.45      # umbral IoU para NMS
TARGET_CLASS    = 0         # clase COCO: 0 = person
INTERVAL_SEC    = 1.0       # ventana temporal (segundos)
LAMBDA_LATENCY  = 1.0       # sensibilidad de latencia (λ)

# Pesos del índice de degradación (wb, wn, wc, wi, wt) — suma = 1
WEIGHTS = dict(blur=0.2, noise=0.2, contrast=0.2, illumination=0.2, temporal=0.2)


# ─────────────────────────────────────────────────────────────
# FUNCIONES DE CALIDAD DE VIDEO
# ─────────────────────────────────────────────────────────────

def blur_degradation(gray):
    """
    Degradación por blur usando varianza del Laplaciano.
    Mayor varianza → imagen más nítida → menor degradación.
    """
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Normalizamos: saturamos en 500 (valor típico para imágenes nítidas)
    score = lap_var / 500.0
    score = min(score, 1.0)
    return 1.0 - score   # invertir: blur alto → degradación alta


def noise_degradation(gray):
    """
    Estimación de ruido: diferencia entre imagen original y mediana filtrada.
    """
    median = cv2.medianBlur(gray, 5).astype(np.float64)
    diff   = np.abs(gray.astype(np.float64) - median)
    noise  = diff.mean()
    # Normalizamos: saturamos en 20 (nivel de ruido alto típico)
    return min(noise / 20.0, 1.0)


def contrast_degradation(gray):
    """
    Degradación de contraste basada en desviación estándar de intensidad.
    Mayor std → más contraste → menos degradación.
    """
    std = gray.std()
    score = std / 80.0   # 80 es std típico de imagen con buen contraste
    score = min(score, 1.0)
    return 1.0 - score


def illumination_degradation(gray):
    """
    Degradación de iluminación basada en brillo promedio.
    Valores extremos (muy oscuro o muy brillante) → mayor degradación.
    """
    mean_brightness = gray.mean()
    # Penalizar desviación del centro (128)
    degradation = abs(mean_brightness - 128.0) / 128.0
    return min(degradation, 1.0)


def temporal_instability(gray_curr, gray_prev):
    """
    Inestabilidad temporal: diferencia frame a frame.
    Artefactos de transmisión producen cambios bruscos.
    """
    if gray_prev is None:
        return 0.0
    diff = np.abs(gray_curr.astype(np.float64) - gray_prev.astype(np.float64))
    instability = diff.mean()
    return min(instability / 30.0, 1.0)


def compute_degradation_index(db, dn, dc, di, dt, w=WEIGHTS):
    """
    Ecuación (3) del paper:
    D_video(t) = wb*Db + wn*Dn + wc*Dc + wi*Di + wt*Dt
    """
    return (w['blur']         * db +
            w['noise']        * dn +
            w['contrast']     * dc +
            w['illumination'] * di +
            w['temporal']     * dt)


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────

def process_video(video_path, model_path='yolov8n.pt', output_csv='resultados.csv'):
    """
    Procesa un video FPV completo, extrae métricas por intervalo de 1 segundo
    y guarda los resultados en un CSV.
    """
    print(f"\n{'='*60}")
    print(f"Video: {video_path}")
    print(f"Modelo: {model_path}")
    print(f"{'='*60}\n")

    # Cargar modelo YOLO
    model = YOLO(model_path)

    # Abrir video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS)
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_fr / fps if fps > 0 else 0
    print(f"FPS: {fps:.1f} | Frames totales: {total_fr} | Duración: {duration:.1f}s\n")

    frames_per_interval = max(1, int(fps * INTERVAL_SEC))

    records    = []
    frame_idx  = 0
    interval   = 0
    gray_prev  = None

    # Buffers por intervalo
    confidences   = []
    db_list, dn_list, dc_list, di_list, dt_list = [], [], [], [], []
    latencies     = []
    n_detections  = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Métricas de calidad ──────────────────────────────
        db = blur_degradation(gray)
        dn = noise_degradation(gray)
        dc = contrast_degradation(gray)
        di = illumination_degradation(gray)
        dt = temporal_instability(gray, gray_prev)

        db_list.append(db); dn_list.append(dn); dc_list.append(dc)
        di_list.append(di); dt_list.append(dt)

        # ── Detección YOLO ───────────────────────────────────
        t0 = time.perf_counter()
        results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD,
                        classes=[TARGET_CLASS], verbose=False)
        latency = time.perf_counter() - t0

        latencies.append(latency)

        # Extraer confianzas de detecciones de persona
        frame_confs = []
        for r in results:
            for box in r.boxes:
                if int(box.cls) == TARGET_CLASS:
                    frame_confs.append(float(box.conf))

        confidences.append(frame_confs)
        n_detections.append(len(frame_confs))

        gray_prev = gray
        frame_idx += 1

        # ── Al completar el intervalo, calcular métricas ─────
        if frame_idx % frames_per_interval == 0:
            t_sec = interval * INTERVAL_SEC

            # Confianza media (Ec. 1 del paper)
            all_confs = [c for fc in confidences for c in fc]
            c_yolo = float(np.mean(all_confs)) if all_confs else 0.0

            # Degradación media del intervalo (Ec. 3)
            d_video = compute_degradation_index(
                np.mean(db_list), np.mean(dn_list), np.mean(dc_list),
                np.mean(di_list), np.mean(dt_list)
            )

            # Latencia media
            L = float(np.mean(latencies))

            # Detecciones correctas (frames con ≥1 detección) vs esperadas
            n_correct  = sum(1 for n in n_detections if n > 0)
            n_expected = len(n_detections)

            # Confiabilidad experimental (Ec. 25)
            R_exp = n_correct / n_expected if n_expected > 0 else 0.0

            records.append({
                'time_s'          : t_sec,
                'interval'        : interval,
                'c_yolo'          : round(c_yolo, 4),
                'd_blur'          : round(np.mean(db_list), 4),
                'd_noise'         : round(np.mean(dn_list), 4),
                'd_contrast'      : round(np.mean(dc_list), 4),
                'd_illumination'  : round(np.mean(di_list), 4),
                'd_temporal'      : round(np.mean(dt_list), 4),
                'd_video'         : round(d_video, 4),
                'latency_s'       : round(L, 4),
                'n_correct'       : n_correct,
                'n_expected'      : n_expected,
                'R_exp'           : round(R_exp, 4),
            })

            print(f"t={t_sec:.0f}s | C_YOLO={c_yolo:.3f} | "
                  f"D_video={d_video:.3f} | L={L*1000:.1f}ms | R_exp={R_exp:.3f}")

            # Resetear buffers
            confidences  = []
            db_list = dn_list = dc_list = di_list = dt_list = []
            latencies    = []
            n_detections = []
            interval    += 1

    cap.release()

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Resultados guardados en: {output_csv}")
    print(df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detección FPV + métricas de calidad')
    parser.add_argument('--video',  required=True,              help='Ruta al video FPV')
    parser.add_argument('--model',  default='yolov8n.pt',       help='Modelo YOLO (.pt)')
    parser.add_argument('--output', default='resultados.csv',   help='CSV de salida')
    args = parser.parse_args()

    process_video(args.video, args.model, args.output)