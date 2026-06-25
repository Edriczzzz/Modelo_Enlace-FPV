"""
validacion.py
=============
Script 2: Cálculo del modelo matemático de confiabilidad + validación estadística + gráficas
Basado en el paper:
  "A Communication-Perception Reliability Model for YOLO-Based Human Detection in UAV FPV Systems"
  R. Vera-Amaro, IPN UPIITA

USO:
    # Con un solo CSV:
    python validacion.py --csvs video1.csv --labels "Video 1"

    # Con múltiples CSVs (un CSV por video):
    python validacion.py --csvs video1.csv video2.csv video3.csv \
                         --labels "Video 1 (Analógico)" "Video 2 (Analógico)" "Video 3 (Digital)" \
                         --lambda_val 1.0

REQUISITOS:
    pip install numpy pandas scipy matplotlib
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import stats
import os


# ─────────────────────────────────────────────────────────────
# ESTILO DE GRÁFICAS (igual al paper pero con texto más grande)
# ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size'        : 13,
    'axes.titlesize'   : 14,
    'axes.labelsize'   : 13,
    'xtick.labelsize'  : 12,
    'ytick.labelsize'  : 12,
    'legend.fontsize'  : 12,
    'lines.linewidth'  : 2.0,
    'lines.markersize' : 8,
    'figure.dpi'       : 150,
    'axes.grid'        : True,
    'grid.alpha'       : 0.4,
})

COLORES  = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
MARKERS  = ['o', 's', '^', 'D', 'v']


# ─────────────────────────────────────────────────────────────
# MODELO MATEMÁTICO (Ecuaciones del paper)
# ─────────────────────────────────────────────────────────────

def modelo_M1(d_video):
    """M1: solo degradación — Ec. (26)"""
    return 1.0 - d_video

def modelo_M2(c_yolo, d_video):
    """M2: confianza + degradación — Ec. (27)"""
    return c_yolo * (1.0 - d_video)

def modelo_M3(c_yolo, d_video, latency, lam=1.0):
    """
    M3: modelo completo — Ec. (17) / (28)
    R_sys(t) = C_YOLO(t) * [1 - D_video(t)] * exp(-λ * L(t))
    """
    return c_yolo * (1.0 - d_video) * np.exp(-lam * latency)

def confiabilidad_latencia(latency, lam=1.0):
    """R_lat = exp(-λ*L) — Ec. (13)"""
    return np.exp(-lam * latency)


# ─────────────────────────────────────────────────────────────
# VALIDACIÓN ESTADÍSTICA
# ─────────────────────────────────────────────────────────────

def estadisticas(y_exp, y_model, nombre='Modelo'):
    """Calcula Pearson r, R², MAE, RMSE."""
    y_exp   = np.array(y_exp)
    y_model = np.array(y_model)

    r, p_val = stats.pearsonr(y_exp, y_model)
    ss_res   = np.sum((y_exp - y_model)**2)
    ss_tot   = np.sum((y_exp - np.mean(y_exp))**2)
    r2       = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    mae      = np.mean(np.abs(y_exp - y_model))
    rmse     = np.sqrt(np.mean((y_exp - y_model)**2))

    print(f"\n{'─'*45}")
    print(f"  Estadísticas — {nombre}")
    print(f"{'─'*45}")
    print(f"  Pearson r  : {r:.4f}  (p={p_val:.2e})")
    print(f"  R²         : {r2:.4f}")
    print(f"  MAE        : {mae:.4f}")
    print(f"  RMSE       : {rmse:.4f}")

    return {'r': r, 'r2': r2, 'mae': mae, 'rmse': rmse}


# ─────────────────────────────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────────────────────────────

def grafica_temporal(dfs, labels, lam, out_dir):
    """Fig. 9 del paper: Validación temporal por video."""
    for i, (df, label) in enumerate(zip(dfs, labels)):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(df['time_s'], df['R_exp'],   'o-', color='steelblue',
                label='Experimental', linewidth=2, markersize=7)
        ax.plot(df['time_s'], df['R_model'], 's--', color='darkorange',
                label='Modelo final', linewidth=2, markersize=7)
        ax.set_xlabel('Tiempo [s]', fontsize=13)
        ax.set_ylabel('$R_{sys}$', fontsize=13)
        ax.set_title(f'Validación temporal — {label}', fontsize=14)
        ax.legend(fontsize=12)
        ax.set_ylim(-0.05, max(df[['R_exp','R_model']].max()) * 1.15 + 0.05)
        fig.tight_layout()
        fname = os.path.join(out_dir, f'temporal_{i+1}.png')
        fig.savefig(fname, dpi=150)
        print(f"  Guardada: {fname}")
        plt.close(fig)


def grafica_degradacion_vs_confiabilidad(dfs, labels, out_dir):
    """Fig. 6 del paper: R_exp vs D_video."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        ax.scatter(df['d_video'], df['R_exp'],
                   color=COLORES[i % len(COLORES)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=label, s=70, alpha=0.85)
    ax.set_xlabel('Degradación observable $D_{video}$', fontsize=13)
    ax.set_ylabel('Confiabilidad experimental $R^{exp}_{det}$', fontsize=13)
    ax.set_title('Confiabilidad de detección vs Degradación de video', fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fname = os.path.join(out_dir, 'degradacion_vs_confiabilidad.png')
    fig.savefig(fname, dpi=150)
    print(f"  Guardada: {fname}")
    plt.close(fig)


def grafica_confianza_vs_confiabilidad(dfs, labels, out_dir):
    """Fig. 7 del paper: R_exp vs C_YOLO."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (df, label) in enumerate(zip(dfs, labels)):
        ax.scatter(df['c_yolo'], df['R_exp'],
                   color=COLORES[i % len(COLORES)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=label, s=70, alpha=0.85)
    ax.set_xlabel('Confianza media YOLO $C_{YOLO}$', fontsize=13)
    ax.set_ylabel('Confiabilidad experimental $R_{sys}$', fontsize=13)
    ax.set_title('Confiabilidad del sistema vs Confianza YOLO', fontsize=14)
    ax.legend(fontsize=12)
    fig.tight_layout()
    fname = os.path.join(out_dir, 'confianza_vs_confiabilidad.png')
    fig.savefig(fname, dpi=150)
    print(f"  Guardada: {fname}")
    plt.close(fig)


def grafica_modelos_comparacion(all_exp, all_M1, all_M2, all_M3, out_dir):
    """Fig. 8 del paper: comparación de modelos candidatos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(all_exp, all_M1, color='#1f77b4', marker='o',
               label='M1: $D_{video}$ solo', s=60, alpha=0.75)
    ax.scatter(all_exp, all_M2, color='#ff7f0e', marker='s',
               label='M2: $D_{video}$ + confianza', s=60, alpha=0.75)
    ax.scatter(all_exp, all_M3, color='#2ca02c', marker='^',
               label='M3: Modelo completo (con latencia)', s=60, alpha=0.85)
    lim = max(max(all_exp), max(all_M3)) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', linewidth=1.5, label='Ideal')
    ax.set_xlabel('$R^{exp}_{sys}$ (Experimental)', fontsize=13)
    ax.set_ylabel('$R^{model}_{sys}$ (Modelo)', fontsize=13)
    ax.set_title('Comparación de modelos candidatos', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, lim)
    ax.set_ylim(-0.02, lim)
    fig.tight_layout()
    fname = os.path.join(out_dir, 'comparacion_modelos.png')
    fig.savefig(fname, dpi=150)
    print(f"  Guardada: {fname}")
    plt.close(fig)


def grafica_bland_altman(all_exp, all_model, labels_pts, out_dir):
    """Fig. 10 del paper: Bland-Altman."""
    all_exp   = np.array(all_exp)
    all_model = np.array(all_model)
    means     = (all_exp + all_model) / 2.0
    diffs     = all_exp - all_model
    bias      = np.mean(diffs)
    sd        = np.std(diffs)
    loa_up    = bias + 1.96 * sd
    loa_low   = bias - 1.96 * sd

    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = list(dict.fromkeys(labels_pts))
    for lbl in unique_labels:
        idx = [i for i, l in enumerate(labels_pts) if l == lbl]
        col_idx = unique_labels.index(lbl)
        ax.scatter(means[idx], diffs[idx],
                   color=COLORES[col_idx % len(COLORES)],
                   marker=MARKERS[col_idx % len(MARKERS)],
                   label=lbl, s=65, alpha=0.85)

    ax.axhline(bias,   color='steelblue',  linestyle='--', linewidth=2,
               label=f'Sesgo medio = {bias:.4f}')
    ax.axhline(loa_up, color='gray', linestyle=':', linewidth=1.8,
               label=f'+1.96 SD = {loa_up:.4f}')
    ax.axhline(loa_low,color='gray', linestyle=':', linewidth=1.8,
               label=f'−1.96 SD = {loa_low:.4f}')
    ax.set_xlabel('Media de $R^{exp}_{sys}$ y $R^{model}_{sys}$', fontsize=13)
    ax.set_ylabel('Diferencia $R^{exp}_{sys} - R^{model}_{sys}$', fontsize=13)
    ax.set_title('Análisis Bland–Altman', fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fname = os.path.join(out_dir, 'bland_altman.png')
    fig.savefig(fname, dpi=150)
    print(f"  Guardada: {fname}")
    plt.close(fig)


def grafica_tabla_estadisticas(stats_dict, out_dir):
    """Tabla resumen de métricas estadísticas."""
    fig, ax = plt.subplots(figsize=(6, 2.5))
    ax.axis('off')
    data = [
        ['Métrica', 'Valor'],
        ['Pearson r',  f"{stats_dict['r']:.4f}"],
        ['R²',         f"{stats_dict['r2']:.4f}"],
        ['MAE',        f"{stats_dict['mae']:.4f}"],
        ['RMSE',       f"{stats_dict['rmse']:.4f}"],
    ]
    tabla = ax.table(cellText=data[1:], colLabels=data[0],
                     cellLoc='center', loc='center',
                     colWidths=[0.5, 0.4])
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(12)
    tabla.scale(1.2, 1.6)
    ax.set_title('Resultados de validación estadística', fontsize=13, pad=10)
    fig.tight_layout()
    fname = os.path.join(out_dir, 'tabla_estadisticas.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    print(f"  Guardada: {fname}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Validación del modelo de confiabilidad')
    parser.add_argument('--csvs',       nargs='+', required=True,
                        help='Archivos CSV generados por detection_performance.py')
    parser.add_argument('--labels',     nargs='+', required=True,
                        help='Etiquetas para cada video (mismo orden que --csvs)')
    parser.add_argument('--lambda_val', type=float, default=1.0,
                        help='Coeficiente λ de sensibilidad a latencia (default=1.0)')
    parser.add_argument('--out_dir',    default='graficas',
                        help='Carpeta de salida para las gráficas')
    args = parser.parse_args()

    if len(args.csvs) != len(args.labels):
        raise ValueError("El número de --csvs debe coincidir con el número de --labels")

    os.makedirs(args.out_dir, exist_ok=True)
    lam = args.lambda_val

    dfs         = []
    all_exp     = []
    all_M1      = []
    all_M2      = []
    all_M3      = []
    labels_pts  = []   # para Bland-Altman

    print(f"\n{'='*60}")
    print("  MODELO DE CONFIABILIDAD — VALIDACIÓN")
    print(f"  λ = {lam}")
    print(f"{'='*60}")

    for csv_path, label in zip(args.csvs, args.labels):
        df = pd.read_csv(csv_path)

        # Calcular modelos
        df['R_M1']   = modelo_M1(df['d_video'])
        df['R_M2']   = modelo_M2(df['c_yolo'], df['d_video'])
        df['R_model'] = modelo_M3(df['c_yolo'], df['d_video'], df['latency_s'], lam)
        df['R_lat']  = confiabilidad_latencia(df['latency_s'], lam)

        dfs.append(df)

        all_exp    += df['R_exp'].tolist()
        all_M1     += df['R_M1'].tolist()
        all_M2     += df['R_M2'].tolist()
        all_M3     += df['R_model'].tolist()
        labels_pts += [label] * len(df)

        print(f"\n  Video: {label}")
        print(df[['time_s','c_yolo','d_video','latency_s',
                  'R_exp','R_model']].to_string(index=False))

    # ── Estadísticas globales ─────────────────────────────────
    stats_res = estadisticas(all_exp, all_M3, 'Modelo M3 (global)')
    estadisticas(all_exp, all_M1, 'Modelo M1 (solo degradación)')
    estadisticas(all_exp, all_M2, 'Modelo M2 (degradación + confianza)')

    # ── Gráficas ──────────────────────────────────────────────
    print(f"\n  Generando gráficas en '{args.out_dir}/'...")
    grafica_temporal(dfs, args.labels, lam, args.out_dir)
    grafica_degradacion_vs_confiabilidad(dfs, args.labels, args.out_dir)
    grafica_confianza_vs_confiabilidad(dfs, args.labels, args.out_dir)
    grafica_modelos_comparacion(all_exp, all_M1, all_M2, all_M3, args.out_dir)
    grafica_bland_altman(all_exp, all_M3, labels_pts, args.out_dir)
    grafica_tabla_estadisticas(stats_res, args.out_dir)

    print(f"\n✅ Todo listo. Revisa la carpeta '{args.out_dir}/'")


if __name__ == '__main__':
    main()