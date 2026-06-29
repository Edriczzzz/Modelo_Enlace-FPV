"""
visualizacion_live.py
=====================
Reproduce un video FPV mostrando en tiempo real:
  - Detecciones YOLO con bounding boxes y confianza
  - Variables del modelo: C_YOLO, D_video, R_exp (experimental), R_mod (modelo M2), Latencia
  - Mini barras de confiabilidad visual (panel compacto, no tapa el video)

Alineado con detection_performance.py / validacion.py:
  - D_video_obs usa exactamente los mismos pesos y funciones que detection_performance.py
    (blur 0.30, contrast 0.20, brightness 0.15, noise 0.20, temporal 0.15)
  - R_latency es binario por umbral (latencia <= LATENCY_THRESHOLD_MS), igual que en
    detection_performance.py / validacion.py (no exponencial)
  - R_exp (experimental) = promedio de R_system_proxy por frame en el intervalo,
    igual que la columna "R_system_proxy" del CSV
  - R_mod (modelo) = C_YOLO * (1 - D_video) * R_latency  → modelo M2 final de validacion.py

USO:
    python visualizacion_live.py --video ruta/al/video.mp4 --model yolov8n.pt --tipo Digital

CONTROLES:
    ESPACIO = pausar/reanudar
    Q       = salir
    G       = guardar frame actual como imagen
    P       = mostrar/ocultar panel

REQUISITOS:
    pip install ultralytics opencv-python numpy
"""

import cv2
import numpy as np
import time
import argparse
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
# PARÁMETROS  (mismos valores que detection_performance.py)
# ─────────────────────────────────────────────────────────────
TARGET_CLASSES        = ["person"]
CONF_THRESHOLD         = 0.35
IOU_THRESHOLD          = 0.45
LATENCY_THRESHOLD_MS   = 200
INTERVAL_SEC           = 1.0

# Colores BGR
COLOR_BOX      = (0, 255, 0)
COLOR_TITULO   = (255, 255, 255)
COLOR_VERDE    = (80, 200, 80)
COLOR_AMARILLO = (0, 210, 210)
COLOR_ROJO     = (60, 60, 220)
COLOR_CYAN     = (200, 200, 0)


# ─────────────────────────────────────────────────────────────
# MÉTRICAS DE CALIDAD  (idénticas a detection_performance.py)
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# PANEL COMPACTO  (texto chico + mini-barras, sin caja de fondo)
# ─────────────────────────────────────────────────────────────
BAR_W = 70     # antes 110/160
BAR_H = 6      # antes 9/14
ESC   = 0.40   # antes 0.48/0.55
GR    = 1
LINE_H = 14    # espaciado vertical entre líneas

def txt_sombra(img, t, x, y, color, esc=ESC, gr=GR):
    cv2.putText(img, t, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, esc, (0, 0, 0), gr+1, cv2.LINE_AA)
    cv2.putText(img, t, (x, y),     cv2.FONT_HERSHEY_SIMPLEX, esc, color,     gr,   cv2.LINE_AA)

def mini_barra(img, x, y, val, color, w=BAR_W, h=BAR_H):
    cv2.rectangle(img, (x, y), (x+w, y+h), (40, 40, 40), -1)
    fill = int(w * min(max(val, 0.0), 1.0))
    if fill > 0:
        cv2.rectangle(img, (x, y), (x+fill, y+h), color, -1)
    cv2.rectangle(img, (x, y), (x+w, y+h), (100, 100, 100), 1)

def dibujar_panel(frame, v):
    px, py = 8, 14

    txt_sombra(frame, f"Det:{v['n_det']} t={v['time_s']:.1f}s", px, py, COLOR_CYAN)
    py += LINE_H

    c = v['c_yolo']
    cc = COLOR_VERDE if c > 0.6 else (COLOR_AMARILLO if c > 0.3 else COLOR_ROJO)
    txt_sombra(frame, f"C_YOLO {c:.2f}", px, py, cc)
    mini_barra(frame, px+78, py-6, c, cc)
    py += LINE_H

    dv = v['d_video']
    cd = COLOR_ROJO if dv > 0.5 else (COLOR_AMARILLO if dv > 0.3 else COLOR_VERDE)
    txt_sombra(frame, f"D_video {dv:.2f}", px, py, cd)
    mini_barra(frame, px+78, py-6, dv, cd)
    py += LINE_H

    lat_ms = v['latency_ms']
    cl = COLOR_VERDE if lat_ms < 80 else (COLOR_AMARILLO if lat_ms < 150 else COLOR_ROJO)
    txt_sombra(frame, f"Lat {lat_ms:.0f}ms", px, py, cl)
    mini_barra(frame, px+78, py-6, min(lat_ms/LATENCY_THRESHOLD_MS, 1.0), cl)
    py += LINE_H

    r_exp = v['r_exp']
    ce = COLOR_VERDE if r_exp > 0.5 else (COLOR_AMARILLO if r_exp > 0.2 else COLOR_ROJO)
    txt_sombra(frame, f"R_exp {r_exp:.2f}", px, py, ce, esc=0.42, gr=1)
    mini_barra(frame, px+78, py-6, r_exp, ce)
    py += LINE_H

    r_mod = v['r_mod']
    cm = COLOR_VERDE if r_mod > 0.5 else (COLOR_AMARILLO if r_mod > 0.2 else COLOR_ROJO)
    txt_sombra(frame, f"R_mod {r_mod:.2f}", px, py, cm, esc=0.42, gr=1)
    mini_barra(frame, px+78, py-6, r_mod, cm)
    py += LINE_H

    tipo = v.get('tipo', '')
    if tipo:
        ct = (0, 200, 255) if 'igital' in tipo else (255, 180, 0)
        txt_sombra(frame, tipo, px, py, ct, esc=0.38)

    return frame


# ─────────────────────────────────────────────────────────────
# FUNCIÓN PRINCIPAL
# ─────────────────────────────────────────────────────────────
def visualizar(video_path, model_path='yolov8n.pt', tipo=''):
    model       = YOLO(model_path)
    class_names = model.names
    target_ids  = [cid for cid, name in class_names.items() if name in TARGET_CLASSES]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir: {video_path}")

    fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\nVideo: {video_path}")
    print(f"FPS: {fps:.1f} | Frames: {total_fr} | Duración: {total_fr/fps:.1f}s")
    print("Controles: ESPACIO=pausar  Q=salir  G=guardar frame  P=mostrar/ocultar panel\n")

    frames_per_interval = max(1, int(fps * INTERVAL_SEC))
    gray_prev = None
    frame_idx = 0
    pausado   = False
    guardados = 0
    panel_on  = True

    # Buffers del intervalo actual (estilo window_metrics.csv)
    buf_dvideo  = []
    buf_detsucc = []
    buf_maxconf = []
    buf_latsucc = []
    buf_latms   = []
    buf_rsysobs = []   # R_system_proxy por frame (igual que detection_performance.py)

    vars_intervalo = {
        'c_yolo': 0.0, 'd_video': 0.0, 'latency_ms': 0.0,
        'r_exp': 0.0, 'r_mod': 0.0, 'n_det': 0, 'time_s': 0.0, 'tipo': tipo,
    }

    frame = None
    while True:
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                print("Fin del video.")
                break

            frame_idx += 1
            t_sec = frame_idx / fps

            h, w = frame.shape[:2]
            max_w = 1100
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            gray = compute_gray(frame)

            # ── Métricas de calidad por frame (igual a detection_performance.py) ──
            brightness           = compute_brightness(gray)
            contrast             = compute_contrast(gray)
            blur                 = compute_blur(gray)
            noise                = compute_noise(gray)
            temporal_instability = compute_temporal_instability(gray_prev, gray)
            d_video_obs          = compute_video_degradation(blur, contrast, brightness, noise, temporal_instability)

            # ── Detección YOLO + latencia ──
            t0 = time.perf_counter()
            results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)
            total_lat_ms = (time.perf_counter() - t0) * 1000

            frame_confs = []
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in target_ids:
                        conf = float(box.conf[0])
                        frame_confs.append(conf)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BOX, 2)
                        etiqueta = f"person {conf:.2f}"
                        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(frame, (x1, y1-20), (x1+tw+6, y1), COLOR_BOX, -1)
                        cv2.putText(frame, etiqueta, (x1+3, y1-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            detection_success = 1 if frame_confs else 0
            max_confidence    = float(np.max(frame_confs)) if frame_confs else 0.0
            lat_success       = 1 if total_lat_ms <= LATENCY_THRESHOLD_MS else 0
            r_det_frame       = detection_success * max_confidence
            r_sys_frame       = (1.0 - d_video_obs) * r_det_frame * lat_success  # = R_system_proxy

            buf_dvideo.append(d_video_obs)
            buf_detsucc.append(detection_success)
            buf_maxconf.append(max_confidence)
            buf_latsucc.append(lat_success)
            buf_latms.append(total_lat_ms)
            buf_rsysobs.append(r_sys_frame)
            gray_prev = gray

            # ── Cada INTERVAL_SEC, recalcular variables agregadas (estilo window_metrics) ──
            if frame_idx % frames_per_interval == 0:
                D_video_mean = float(np.mean(buf_dvideo))
                C_YOLO       = float(np.mean(buf_maxconf))
                R_latency    = float(np.mean(buf_latsucc))
                lat_mean_ms  = float(np.mean(buf_latms))
                R_exp        = float(np.mean(buf_rsysobs))          # experimental (R_system_proxy)
                R_det_M2     = C_YOLO * (1.0 - D_video_mean)
                R_mod        = R_det_M2 * R_latency                  # modelo final M2

                vars_intervalo.update({
                    'c_yolo': C_YOLO, 'd_video': D_video_mean,
                    'latency_ms': lat_mean_ms, 'r_exp': R_exp, 'r_mod': R_mod,
                    'n_det': len(frame_confs), 'time_s': t_sec,
                })
                buf_dvideo, buf_detsucc, buf_maxconf = [], [], []
                buf_latsucc, buf_latms, buf_rsysobs = [], [], []
            else:
                vars_intervalo['n_det']  = len(frame_confs)
                vars_intervalo['time_s'] = t_sec

        frame_vis = frame.copy()
        if panel_on:
            frame_vis = dibujar_panel(frame_vis, vars_intervalo)

        if pausado:
            txt_sombra(frame_vis, "[ PAUSADO ]", 8, frame_vis.shape[0]-12, COLOR_AMARILLO, esc=0.6, gr=2)

        cv2.imshow("UAV-FPV | Deteccion YOLO + Modelo Confiabilidad", frame_vis)

        delay = 1 if not pausado else 50
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('q') or key == 27:
            print("Saliendo...")
            break
        elif key == ord(' '):
            pausado = not pausado
            print("Pausado" if pausado else "Reanudado")
        elif key == ord('g'):
            guardados += 1
            nombre = f"frame_captura_{guardados:03d}.png"
            cv2.imwrite(nombre, frame_vis)
            print(f"  Frame guardado: {nombre}")
        elif key == ord('p'):
            panel_on = not panel_on

    cap.release()
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualización en tiempo real YOLO + Modelo (alineado con detection_performance.py)')
    parser.add_argument('--video', required=True,        help='Ruta al video FPV')
    parser.add_argument('--model', default='yolov8n.pt', help='Modelo YOLO (.pt)')
    parser.add_argument('--tipo',  default='',           help='Etiqueta del video (ej: "Analogico" o "Digital")')
    args = parser.parse_args()

    visualizar(args.video, args.model, args.tipo)