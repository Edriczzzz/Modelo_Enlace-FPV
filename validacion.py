"""
validacion.py
=============
Validación del modelo de confiabilidad con múltiples videos analógicos y digitales.
Adaptado del script original del Dr. Vera-Amaro.

USO:
    python validacion.py

    Edita INPUT_FILES abajo para apuntar a tus window_metrics.csv de cada video.

REQUISITOS:
    pip install numpy pandas scipy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ============================================================
# CONFIGURACIÓN — edita esto con tus archivos
# ============================================================

INPUT_FILES = {
    # ── Analógicos ──────────────────────────────────────────
    "Analógico V1": "resultados_modelo_fpv/video_analog_1/window_metrics.csv",
    "Analógico V2": "resultados_modelo_fpv/video_analog_2/window_metrics.csv",
    "Analógico V3": "resultados_modelo_fpv/video_analog_3/window_metrics.csv",
    "Analógico V4": "resultados_modelo_fpv/video_analog_4/window_metrics.csv",
    "Analógico V5": "resultados_modelo_fpv/video_analog_5/window_metrics.csv",
    # ── Digitales ───────────────────────────────────────────
    "Digital D1":   "resultados_modelo_fpv/video_digital_1/window_metrics.csv",
    "Digital D2":   "resultados_modelo_fpv/video_digital_2/window_metrics.csv",
    "Digital D3":   "resultados_modelo_fpv/video_digital_3/window_metrics.csv",
    "Digital D4":   "resultados_modelo_fpv/video_digital_4/window_metrics.csv",
    "Digital D5":   "resultados_modelo_fpv/video_digital_5/window_metrics.csv",
}

OUTPUT_DIR = "validacion_modelo_fpv_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Colores y marcadores para gráficas
COLORES  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
            "#17becf","#bcbd22","#e377c2","#8c564b","#7f7f7f"]
MARKERS  = ["o","s","^","D","v","P","X","*","h","8"]


# ============================================================
# MÉTRICAS ESTADÍSTICAS
# ============================================================

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return np.nan if ss_tot == 0 else 1 - ss_res/ss_tot

def pearson(y_true, y_pred):
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return np.corrcoef(y_true, y_pred)[0,1]

def normalize_series(s):
    s = s.astype(float)
    rng = s.max() - s.min()
    return pd.Series(np.zeros(len(s)), index=s.index) if rng == 0 else (s - s.min()) / rng


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_all_videos():
    required = [
        "window_id","start_time_s","D_video_mean","R_det_exp",
        "max_confidence_mean","R_latency","R_system_proxy",
        "blur_mean","noise_mean","temporal_instability_mean","latency_mean_ms",
    ]
    dfs = []
    for label, path in INPUT_FILES.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"No existe: {path}\n"
                "  → Corre primero detection_performance.py para cada video.")
        df = pd.read_csv(path)
        df["label"] = label
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Faltan columnas en {path}: {missing}")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "combined_raw.csv"), index=False)
    return combined


# ============================================================
# MODELOS  (igual que el script del profe + latencia exponencial)
# ============================================================

def add_models(df):
    df = df.copy()

    # M1: solo degradación observable
    df["R_sys_M1"] = (1.0 - df["D_video_mean"]) * df["R_latency"]

    # M2 final: C_YOLO * (1 - D_video) * R_lat  ← modelo propuesto del paper
    df["R_det_M2"]  = df["max_confidence_mean"] * (1.0 - df["D_video_mean"])
    df["R_sys_M2"]  = df["R_det_M2"] * df["R_latency"]

    # M3: degradación visual ponderada (baseline extendido del profe)
    noise_n = normalize_series(df["noise_mean"])
    temp_n  = normalize_series(df["temporal_instability_mean"])
    blur_n  = normalize_series(df["blur_mean"])
    df["D_visual_weighted"] = (0.40*df["D_video_mean"] +
                                0.25*noise_n + 0.25*temp_n + 0.10*blur_n)
    df["R_det_M3"] = df["max_confidence_mean"] * (1.0 - df["D_visual_weighted"])
    df["R_sys_M3"] = df["R_det_M3"] * df["R_latency"]

    return df


# ============================================================
# VALIDACIÓN ESTADÍSTICA
# ============================================================

def evaluate_models(df):
    models = {
        "M1_Dvideo_only"           : "R_sys_M1",
        "M2_final_Dvideo_confidence": "R_sys_M2",
        "M3_weighted_visual"        : "R_sys_M3",
    }
    rows = []
    groups = [("global", df)] + [(lbl, sub) for lbl, sub in df.groupby("label")]
    for scope, subdf in groups:
        y_true = subdf["R_system_proxy"].values
        for mname, col in models.items():
            y_pred = subdf[col].values
            rows.append({
                "scope"     : scope,
                "model"     : mname,
                "N"         : len(subdf),
                "Pearson_r" : pearson(y_true, y_pred),
                "R2"        : r2(y_true, y_pred),
                "MAE"       : mae(y_true, y_pred),
                "RMSE"      : rmse(y_true, y_pred),
                "mean_bias" : np.mean(y_true - y_pred),
                "std_bias"  : np.std(y_true - y_pred),
            })
    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(OUTPUT_DIR, "validation_metrics.csv"), index=False)
    return metrics


# ============================================================
# GRÁFICAS GLOBALES
# ============================================================

def plot_experimental_vs_model(df):
    fig, ax = plt.subplots(figsize=(8,7))
    ax.scatter(df["R_system_proxy"], df["R_sys_M1"],
               alpha=0.5, s=40, label="M1: $D_{video}$ only")
    ax.scatter(df["R_system_proxy"], df["R_sys_M2"],
               alpha=0.8, s=40, label="M2 final: $D_{video}$ + confidence")
    ax.scatter(df["R_system_proxy"], df["R_sys_M3"],
               alpha=0.6, s=40, label="M3: weighted visual")
    lim = max(df["R_system_proxy"].max(), df["R_sys_M2"].max()) * 1.05
    ax.plot([0,lim],[0,lim],"k--", linewidth=1.5, label="Ideal")
    ax.set_xlabel(r"Experimental $R_{sys}^{exp}$", fontsize=13)
    ax.set_ylabel(r"Model $R_{sys}^{model}$", fontsize=13)
    ax.set_title("Experimental vs Model System Reliability\n(Analógico + Digital)", fontsize=13)
    ax.legend(fontsize=11); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"01_experimental_vs_model_all.png"), dpi=300)
    plt.close(fig)


def plot_temporal_by_video(df):
    for i, (label, sub) in enumerate(df.groupby("label", sort=False)):
        fig, ax = plt.subplots(figsize=(10,4))
        ax.plot(sub["start_time_s"], sub["R_system_proxy"],
                "o-", linewidth=2, markersize=6, label="Experimental")
        ax.plot(sub["start_time_s"], sub["R_sys_M2"],
                "s--", linewidth=2, markersize=6, label="Modelo final (M2)")
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel(r"$R_{sys}$", fontsize=12)
        ax.set_title(f"Temporal Validation — {label}", fontsize=13)
        ax.legend(fontsize=11); ax.grid(True); fig.tight_layout()
        safe = label.replace(" ","_").replace("/","_")
        fig.savefig(os.path.join(OUTPUT_DIR,f"02_temporal_{safe}.png"), dpi=300)
        plt.close(fig)


def plot_degradacion_vs_confiabilidad(df):
    fig, ax = plt.subplots(figsize=(8,6))
    # Agrupar por tipo: azul=Analógico, verde=Digital
    tipo_cfg = {"analogico": ("#1f77b4", "o", "Analógico FPV"),
                "digital":   ("#2ca02c", "s", "Digital FPV")}
    for tipo, (color, marker, label) in tipo_cfg.items():
        sub = df[df["tipo"] == tipo]
        if len(sub) == 0:
            continue
        ax.scatter(sub["D_video_mean"], sub["R_det_exp"],
                   color=color, marker=marker, label=label,
                   s=60, alpha=0.75, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(r"Observable Video Degradation $D_{video}^{obs}$", fontsize=13)
    ax.set_ylabel(r"Experimental Detection Reliability $R_{det}^{exp}$", fontsize=13)
    ax.set_title("Detection Reliability vs Video Degradation\n(Analógico vs Digital)", fontsize=13)
    ax.legend(fontsize=12); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"03_Rdet_vs_Dvideo.png"), dpi=300)
    plt.close(fig)


def plot_confianza_vs_confiabilidad(df):
    fig, ax = plt.subplots(figsize=(8,6))
    # Agrupar por tipo: azul=Analógico, verde=Digital
    tipo_cfg = {"analogico": ("#1f77b4", "o", "Analógico FPV"),
                "digital":   ("#2ca02c", "s", "Digital FPV")}
    for tipo, (color, marker, label) in tipo_cfg.items():
        sub = df[df["tipo"] == tipo]
        if len(sub) == 0:
            continue
        ax.scatter(sub["max_confidence_mean"], sub["R_system_proxy"],
                   color=color, marker=marker, label=label,
                   s=60, alpha=0.75, edgecolors="white", linewidths=0.4)
    ax.set_xlabel(r"Mean YOLO Confidence $C_{YOLO}$", fontsize=13)
    ax.set_ylabel(r"Experimental $R_{sys}$", fontsize=13)
    ax.set_title("System Reliability vs YOLO Confidence\n(Analógico vs Digital)", fontsize=13)
    ax.legend(fontsize=12); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"04_Rsys_vs_confidence.png"), dpi=300)
    plt.close(fig)


def plot_bland_altman(df):
    means = (df["R_system_proxy"] + df["R_sys_M2"]) / 2
    diffs = df["R_system_proxy"] - df["R_sys_M2"]
    bias  = diffs.mean(); sd = diffs.std()

    fig, ax = plt.subplots(figsize=(8,6))
    for i, (label, sub) in enumerate(df.groupby("label", sort=False)):
        m = (sub["R_system_proxy"] + sub["R_sys_M2"]) / 2
        d = sub["R_system_proxy"] - sub["R_sys_M2"]
        ax.scatter(m, d, color=COLORES[i % len(COLORES)],
                   marker=MARKERS[i % len(MARKERS)],
                   label=label, s=55, alpha=0.85)
    ax.axhline(bias,         color="steelblue", linestyle="--",
               linewidth=2,  label=f"Mean bias = {bias:.4f}")
    ax.axhline(bias+1.96*sd, color="gray",      linestyle=":",
               linewidth=1.8,label=f"+1.96 SD = {bias+1.96*sd:.4f}")
    ax.axhline(bias-1.96*sd, color="gray",      linestyle=":",
               linewidth=1.8,label=f"−1.96 SD = {bias-1.96*sd:.4f}")
    ax.set_xlabel(r"Mean of $R_{sys}^{exp}$ and $R_{sys}^{model}$", fontsize=13)
    ax.set_ylabel(r"Difference $R_{sys}^{exp} - R_{sys}^{model}$", fontsize=13)
    ax.set_title("Bland–Altman Analysis — Final Model (M2)\n(Analógico vs Digital)", fontsize=13)
    ax.legend(fontsize=9, ncol=2); ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"05_bland_altman.png"), dpi=300)
    plt.close(fig)


def plot_comparacion_analogico_digital(df):
    """Gráfica exclusiva de comparación por tipo de video."""
    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    for ax, col, ylabel, title in [
        (axes[0], "D_video_mean",      r"$D_{video}^{obs}$",
         "Video Degradation by FPV Type"),
        (axes[1], "max_confidence_mean", r"Mean YOLO Confidence $C_{YOLO}$",
         "YOLO Confidence by FPV Type"),
    ]:
        analog  = df[df["tipo"]=="analogico"][col].dropna()
        digital = df[df["tipo"]=="digital"][col].dropna()
        ax.boxplot([analog, digital],
                   patch_artist=True,
                   boxprops=dict(facecolor="lightblue"),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Analogico", "Digital"])
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="y")

    fig.suptitle("Analógico vs Digital FPV — Comparación de métricas clave", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"06_analogico_vs_digital.png"), dpi=300)
    plt.close(fig)


def plot_tabla_estadisticas(metrics):
    """Tabla resumen de las métricas globales del modelo M2."""
    row = metrics[(metrics["scope"]=="global") &
                  (metrics["model"]=="M2_final_Dvideo_confidence")].iloc[0]
    data = [
        ["Pearson r",  f"{row['Pearson_r']:.4f}"],
        ["R²",         f"{row['R2']:.4f}"],
        ["MAE",        f"{row['MAE']:.4f}"],
        ["RMSE",       f"{row['RMSE']:.4f}"],
        ["Mean bias",  f"{row['mean_bias']:.4f}"],
        ["N samples",  f"{int(row['N'])}"],
    ]
    fig, ax = plt.subplots(figsize=(5,3))
    ax.axis("off")
    t = ax.table(cellText=data, colLabels=["Metric","Value"],
                 cellLoc="center", loc="center", colWidths=[0.55,0.4])
    t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1.2,1.6)
    ax.set_title("Statistical Validation — Model M2 (Global)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"07_tabla_estadisticas.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(df):
    df = df.copy()
    df["residual"] = df["R_system_proxy"] - df["R_sys_M2"]

    fig, axes = plt.subplots(1,2,figsize=(13,5))

    axes[0].hist(df["residual"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_xlabel(r"Residual $R_{sys}^{exp}-R_{sys}^{model}$", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("Residual Error Distribution — Model M2", fontsize=12)
    axes[0].grid(True)

    for i, (label, sub) in enumerate(df.groupby("label", sort=False)):
        axes[1].scatter(sub["R_sys_M2"], sub["residual"],
                        color=COLORES[i%len(COLORES)],
                        marker=MARKERS[i%len(MARKERS)],
                        label=label, s=50, alpha=0.8)
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1.5)
    axes[1].set_xlabel(r"Model $R_{sys}^{model}$", fontsize=12)
    axes[1].set_ylabel("Residual error", fontsize=12)
    axes[1].set_title("Residuals vs Model Prediction", fontsize=12)
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(True)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR,"08_residuals.png"), dpi=300)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n========== CARGANDO DATOS ==========")
    df = load_all_videos()

    print("\n========== CALCULANDO MODELOS ==========")
    df = add_models(df)
    df.to_csv(os.path.join(OUTPUT_DIR,"combined_with_predictions.csv"), index=False)

    print("\n========== VALIDACIÓN ESTADÍSTICA ==========")
    metrics = evaluate_models(df)

    # Imprimir resumen global del modelo M2
    global_m2 = metrics[(metrics["scope"]=="global") &
                         (metrics["model"]=="M2_final_Dvideo_confidence")].iloc[0]
    print(f"\n  ── Modelo M2 (propuesto) — Global ──")
    print(f"  Pearson r  : {global_m2['Pearson_r']:.4f}")
    print(f"  R²         : {global_m2['R2']:.4f}")
    print(f"  MAE        : {global_m2['MAE']:.4f}")
    print(f"  RMSE       : {global_m2['RMSE']:.4f}")
    print(f"  Mean bias  : {global_m2['mean_bias']:.4f}")
    print(f"  N          : {int(global_m2['N'])}")

    print("\n  Métricas por video:")
    by_video = metrics[metrics["scope"]!="global"]
    print(by_video[["scope","model","Pearson_r","R2","MAE","RMSE"]].to_string(index=False))

    print("\n========== GENERANDO GRÁFICAS ==========")
    plot_experimental_vs_model(df)
    plot_temporal_by_video(df)
    plot_degradacion_vs_confiabilidad(df)
    plot_confianza_vs_confiabilidad(df)
    plot_bland_altman(df)
    plot_comparacion_analogico_digital(df)
    plot_tabla_estadisticas(metrics)
    plot_residuals(df)

    print(f"\n✅ Todo guardado en '{OUTPUT_DIR}/'")
    print(f"   Archivos CSV y {8} gráficas listas.\n")


if __name__ == "__main__":
    main()