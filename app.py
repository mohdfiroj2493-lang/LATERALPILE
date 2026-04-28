import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ============================================================
# LPILE-STYLE LATERAL PILE APP
# Finite-difference beam-column solution + manual-based p-y models
# Based on the uploaded LPILE technical manual approach.
# ============================================================

# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------
@dataclass
class PileInputs:
    head_condition: str
    base_condition: str
    H: float
    P: float
    M: float
    L: float
    n_ele: int
    D: float
    E: float


@dataclass
class LayeredPoint:
    z: float
    layer: Dict
    pu: float
    y50: float
    k_init: float


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def safe_float(v, default=0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except Exception:
        return default


def get_pile_section(D: float, E: float) -> Tuple[float, float, float]:
    A = math.pi * D**2 / 4.0
    I = math.pi * D**4 / 64.0
    EI = E * I
    return A, I, EI


def validate_layers(layers: List[Dict], pile_length: float) -> List[str]:
    errors = []
    if not layers:
        errors.append("At least one soil layer is required.")
        return errors

    prev_bot = None
    for i, layer in enumerate(layers):
        z_top = safe_float(layer.get("z_top"), None)
        z_bot = safe_float(layer.get("z_bot"), None)
        soil_type = int(safe_float(layer.get("soilType"), 0))

        if z_top is None or z_bot is None:
            errors.append(f"Layer {i+1}: z_top and z_bot are required.")
            continue
        if z_bot <= z_top:
            errors.append(f"Layer {i+1}: z_bot must be greater than z_top.")
        if prev_bot is not None and abs(z_top - prev_bot) > 1e-9:
            errors.append(f"Layer {i+1}: z_top must equal previous z_bot.")
        prev_bot = z_bot

        if soil_type == 1:
            if layer.get("c") is None:
                errors.append(f"Layer {i+1}: clay layer requires c.")
        elif soil_type == 2:
            if layer.get("phi_deg") is None:
                errors.append(f"Layer {i+1}: sand layer requires phi_deg.")
            if layer.get("gamma") is None:
                errors.append(f"Layer {i+1}: sand layer requires gamma.")
        else:
            errors.append(f"Layer {i+1}: soilType must be 1 (clay) or 2 (sand).")

    if abs(safe_float(layers[-1].get("z_bot"), -999.0) - pile_length) > 1e-9:
        errors.append("The bottom of the last layer must equal the pile length.")
    return errors


def get_layer_at_depth(layers: List[Dict], z: float) -> Dict:
    for layer in layers:
        if safe_float(layer["z_top"]) - 1e-12 <= z <= safe_float(layer["z_bot"]) + 1e-12:
            return layer
    return layers[-1]


def build_layer_table(layers: List[Dict]) -> pd.DataFrame:
    rows = []
    for layer in layers:
        rows.append(
            {
                "name": layer.get("name", "Layer"),
                "z_top": safe_float(layer.get("z_top")),
                "z_bot": safe_float(layer.get("z_bot")),
                "gamma": layer.get("gamma"),
                "Cd": layer.get("Cd"),
                "soilType": int(safe_float(layer.get("soilType"), 0)),
                "phi_deg": layer.get("phi_deg"),
                "k": layer.get("k"),
                "c": layer.get("c"),
                "eps50": layer.get("eps50"),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# p-y models from the manual
# ------------------------------------------------------------
def api_sand_coefficients(phi_deg: float) -> Tuple[float, float, float, float, float, float]:
    phi = math.radians(phi_deg)
    alpha = phi / 2.0
    beta = math.radians(45.0) + phi / 2.0
    K0 = 0.4
    KA = math.tan(math.radians(45.0) - phi / 2.0) ** 2
    KP = math.tan(math.radians(45.0) + phi / 2.0) ** 2
    C1 = KP - KA
    C2 = KP - KA
    C3 = KP * KP + K0 * math.tan(phi)
    return alpha, beta, K0, KA, C1, C2, C3


def api_sand_pu(z: float, D: float, gamma_eff: float, phi_deg: float) -> float:
    # Manual API sand: use shallow/deep ultimate resistance, take smaller.
    # Equations align with manual Sec. 3.4.3.
    _, _, _, _, C1, C2, C3 = api_sand_coefficients(phi_deg)
    z = max(z, 1e-6)
    pus = (C1 * z + C2 * D) * gamma_eff * z
    pud = C3 * D * gamma_eff * z
    return max(min(pus, pud), 1.0)


def api_sand_k(phi_deg: float, gamma_eff: float) -> float:
    # Manual default trend from figure: fine sand above/below water.
    # gamma_eff threshold follows earlier implementation convention.
    if gamma_eff < 12225.0:
        # below water table curve in MN/m3 converted to N/m3
        p = phi_deg
        k_mn = 0.0166 * p**3 - 1.5526 * p**2 + 58.43 * p - 769.18
    else:
        p = phi_deg
        k_mn = 0.4168 * p**2 - 8.1254 * p - 83.664
    return max(k_mn, 5.0) * 1e6


def api_sand_p_y(y: float, z: float, D: float, gamma_eff: float, phi_deg: float, cyclic: bool = False) -> float:
    pu = api_sand_pu(z, D, gamma_eff, phi_deg)
    A = 0.9 if cyclic else max(0.9, 3.0 - 0.8 * z / max(D, 1e-9))
    k = api_sand_k(phi_deg, gamma_eff)
    arg = (k * z * y) / max(A * pu, 1e-12)
    return A * pu * math.tanh(arg)


def matlock_soft_clay_pu(z: float, D: float, c: float, gamma_eff: float, J: float = 0.5) -> float:
    z = max(z, 1e-6)
    pu1 = (3.0 + gamma_eff * z / max(c, 1e-9) + J * z / max(D, 1e-9)) * c * D
    pu2 = 9.0 * c * D
    return max(min(pu1, pu2), 1.0)


def matlock_soft_clay_y50(D: float, eps50: float) -> float:
    return max(2.5 * eps50 * D, 1e-8)


def matlock_soft_clay_p_y(y: float, z: float, D: float, c: float, gamma_eff: float, eps50: float, cyclic: bool = False) -> float:
    pu = matlock_soft_clay_pu(z, D, c, gamma_eff)
    y50 = matlock_soft_clay_y50(D, eps50)
    if not cyclic:
        if y <= 8.0 * y50:
            return 0.5 * pu * (y / y50) ** (1.0 / 3.0)
        return pu

    xr = max(6.0 * c * D / max(gamma_eff * D + 1e-12, 1e-12), 2.5 * D)
    if y <= 3.0 * y50:
        return 0.5 * pu * (y / y50) ** (1.0 / 3.0)
    p15 = 0.72 * pu * min(z / max(xr, 1e-9), 1.0)
    if y <= 15.0 * y50:
        return 0.72 * pu + (p15 - 0.72 * pu) * (y - 3.0 * y50) / (12.0 * y50)
    return p15


def stiff_clay_no_water_pu(z: float, D: float, c: float, gamma_eff: float) -> float:
    pu1 = (3.0 + gamma_eff * z / max(c, 1e-9) + 0.5 * z / max(D, 1e-9)) * c * D
    pu2 = 9.0 * c * D
    return max(min(pu1, pu2), 1.0)


def stiff_clay_no_water_p_y(y: float, z: float, D: float, c: float, gamma_eff: float, eps50: float, N_cycles: int = 1) -> float:
    pu = stiff_clay_no_water_pu(z, D, c, gamma_eff)
    y50 = max(2.5 * eps50 * D, 1e-8)
    ys = y50 * (2.0 * y / max(y50, 1e-12)) ** 4 if y <= 16.0 * y50 else y

    # Static inverse form from manual is represented here directly in p(y)
    # using the equivalent closed relationship.
    if N_cycles <= 1:
        if y <= 16.0 * y50:
            return 0.5 * pu * (y / y50) ** 0.25
        return pu

    # Reese stiff clay without free water cyclic expansion
    # Solve static p from y_c = y_s + y50*C*logN with C = 9.6 (p/pu)^4.
    logN = math.log10(max(N_cycles, 1))
    if y <= 0.0:
        return 0.0

    def f(r: float) -> float:
        r = clamp(r, 0.0, 1.0)
        ys_local = y50 * (2.0 * r) ** 4 if r <= 0.5 else 16.0 * y50 * r
        return ys_local + y50 * 9.6 * (r ** 4) * logN - y

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi) * pu


def stiff_clay_with_water_params(z: float, D: float, c: float, gamma_eff: float) -> Tuple[float, float, float, float]:
    pct = 2.0 * c * D + gamma_eff * D * z + 2.83 * c * z
    pcd = 11.0 * c * D
    pc = min(pct, pcd)
    As = 0.2 + 0.4 * math.tanh(0.62 * z / max(D, 1e-9))
    Ac = 0.2 + 0.1 * math.tanh(1.5 * z / max(D, 1e-9))
    return max(pc, 1.0), As, Ac, max(0.005 * D, 1e-8)


def stiff_clay_with_water_p_y(y: float, z: float, D: float, c: float, gamma_eff: float, eps50: float, cyclic: bool = False) -> float:
    pc, As, Ac, _ = stiff_clay_with_water_params(z, D, c, gamma_eff)
    y50 = max(eps50 * D, 1e-8)

    if not cyclic:
        if y <= As * y50:
            return 0.5 * pc * (y / y50) ** 0.5
        if y <= 6.0 * As * y50:
            term = ((y - As * y50) / max(As * y50, 1e-12))
            return 0.5 * pc * (y / y50) ** 0.5 - 0.055 * pc * term ** 1.25
        if y <= 18.0 * As * y50:
            return pc * (0.5 * As) - 0.0625 * pc * ((y - 6.0 * As * y50) / max(y50, 1e-12))
        return pc * (1.225 - 0.75 * As - 0.411 * As)

    yp = 4.1 * Ac * y50
    if y <= 0.45 * yp:
        return Ac * pc * (1.0 - ((1.0 - y / max(0.45 * yp, 1e-12)) ** 2.5))
    if y <= 1.8 * yp:
        return Ac * pc * (0.936 - 0.085 * (y / max(y50, 1e-12)))
    return Ac * pc * 0.102


def compute_py_curve(layer: Dict, z: float, y: float) -> float:
    soil_type = int(safe_float(layer.get("soilType"), 0))
    gamma = safe_float(layer.get("gamma"), 17000.0)
    model = str(layer.get("model", "default")).lower()
    cyclic = bool(layer.get("cyclic", False))
    D = safe_float(layer.get("pile_diameter_ref"), 1.0)

    if soil_type == 2:
        phi = safe_float(layer.get("phi_deg"), 35.0)
        return api_sand_p_y(y, z, D, gamma, phi, cyclic=cyclic)

    c = safe_float(layer.get("c"), 25000.0)
    eps50 = safe_float(layer.get("eps50"), 0.02)
    if model == "stiff_nowater":
        N_cycles = int(safe_float(layer.get("N_cycles"), 1))
        return stiff_clay_no_water_p_y(y, z, D, c, gamma, eps50, N_cycles=N_cycles)
    if model == "stiff_water":
        return stiff_clay_with_water_p_y(y, z, D, c, gamma, eps50, cyclic=cyclic)
    return matlock_soft_clay_p_y(y, z, D, c, gamma, eps50, cyclic=cyclic)


def compute_py_initial(layer: Dict, z: float, D: float) -> Tuple[float, float, float]:
    soil_type = int(safe_float(layer.get("soilType"), 0))
    gamma = safe_float(layer.get("gamma"), 17000.0)
    if soil_type == 2:
        phi = safe_float(layer.get("phi_deg"), 35.0)
        pu = api_sand_pu(z, D, gamma, phi)
        k_init = api_sand_k(phi, gamma) * max(z, 1e-6)
        y50 = max(pu / max(k_init, 1e-12), 1e-8)
        return pu, y50, k_init

    c = safe_float(layer.get("c"), 25000.0)
    eps50 = safe_float(layer.get("eps50"), 0.02)
    model = str(layer.get("model", "default")).lower()
    if model == "stiff_nowater":
        pu = stiff_clay_no_water_pu(z, D, c, gamma)
        y50 = max(2.5 * eps50 * D, 1e-8)
        k_init = 0.5 * pu / max(y50, 1e-12)
        return pu, y50, k_init
    if model == "stiff_water":
        pc, _, _, _ = stiff_clay_with_water_params(z, D, c, gamma)
        y50 = max(eps50 * D, 1e-8)
        k_init = 0.5 * pc / max(y50, 1e-12)
        return pc, y50, k_init
    pu = matlock_soft_clay_pu(z, D, c, gamma)
    y50 = matlock_soft_clay_y50(D, eps50)
    k_init = (0.5 * pu) / max(y50, 1e-12)
    return pu, y50, k_init


# ------------------------------------------------------------
# Finite-difference solver (manual style)
# ------------------------------------------------------------
def build_initial_response(layers: List[Dict], pile: PileInputs) -> List[LayeredPoint]:
    dz = pile.L / pile.n_ele
    points: List[LayeredPoint] = []
    for i in range(pile.n_ele + 1):
        z = i * dz
        layer = dict(get_layer_at_depth(layers, z))
        layer["pile_diameter_ref"] = pile.D
        pu, y50, k_init = compute_py_initial(layer, z, pile.D)
        points.append(LayeredPoint(z=z, layer=layer, pu=pu, y50=y50, k_init=k_init))
    return points


def solve_beam_fd(pile: PileInputs, layers: List[Dict], max_iter: int = 80, tol: float = 1e-7) -> Dict[str, np.ndarray]:
    # Manual finite-difference framework following Chapter 2 style.
    n = pile.n_ele
    dz = pile.L / n
    m = n + 1
    _, I, EI = get_pile_section(pile.D, pile.E)
    z = np.linspace(0.0, pile.L, m)

    # Initial linearized subgrade profile
    response = build_initial_response(layers, pile)
    ksec = np.array([max(p.k_init, 1.0) for p in response], dtype=float)
    y = np.zeros(m, dtype=float)

    # Iterative secant-stiffness method
    for _ in range(max_iter):
        K = np.zeros((m, m), dtype=float)
        F = np.zeros(m, dtype=float)

        # Interior nodes using EI y'''' + P y'' + p = 0, with p≈ksec*y
        for i in range(2, m - 2):
            K[i, i - 2] += EI / dz**4
            K[i, i - 1] += -4.0 * EI / dz**4 + pile.P / dz**2
            K[i, i] += 6.0 * EI / dz**4 - 2.0 * pile.P / dz**2 + ksec[i]
            K[i, i + 1] += -4.0 * EI / dz**4 + pile.P / dz**2
            K[i, i + 2] += EI / dz**4

        # Head boundary conditions
        if pile.head_condition == "free":
            # M(0)=M_head, V(0)=H
            K[0, 0] = 1.0
            K[0, 1] = -2.0
            K[0, 2] = 1.0
            F[0] = pile.M * dz**2 / EI

            K[1, 0] = -5.0
            K[1, 1] = 18.0
            K[1, 2] = -24.0
            K[1, 3] = 14.0
            K[1, 4] = -3.0
            F[1] = 2.0 * pile.H * dz**3 / EI
        else:
            # fixed head: y'(0)=0 and V(0)=H
            K[0, 0] = -3.0
            K[0, 1] = 4.0
            K[0, 2] = -1.0
            F[0] = 0.0

            K[1, 0] = -5.0
            K[1, 1] = 18.0
            K[1, 2] = -24.0
            K[1, 3] = 14.0
            K[1, 4] = -3.0
            F[1] = 2.0 * pile.H * dz**3 / EI

        # Base boundary conditions
        if pile.base_condition == "fixed":
            K[m - 2, m - 3] = 1.0
            K[m - 2, m - 2] = -4.0
            K[m - 2, m - 1] = 3.0
            F[m - 2] = 0.0

            K[m - 1, m - 3] = 1.0
            K[m - 1, m - 2] = -2.0
            K[m - 1, m - 1] = 1.0
            F[m - 1] = 0.0
        else:
            # pinned/free rotation base with zero moment and zero shear
            K[m - 2, m - 3] = 1.0
            K[m - 2, m - 2] = -2.0
            K[m - 2, m - 1] = 1.0
            F[m - 2] = 0.0

            K[m - 1, m - 5] = 3.0
            K[m - 1, m - 4] = -14.0
            K[m - 1, m - 3] = 24.0
            K[m - 1, m - 2] = -18.0
            K[m - 1, m - 1] = 5.0
            F[m - 1] = 0.0

        y_new = np.linalg.solve(K, F)

        # Update secant stiffness from nonlinear p-y curves
        ksec_new = np.zeros_like(ksec)
        for i, zi in enumerate(z):
            layer = dict(get_layer_at_depth(layers, zi))
            layer["pile_diameter_ref"] = pile.D
            yi = abs(y_new[i])
            p = compute_py_curve(layer, zi, yi)
            ksec_new[i] = max(p / max(yi, 1e-8), 1.0)

        if np.max(np.abs(y_new - y)) < tol:
            y = y_new
            ksec = ksec_new
            break

        y = y_new
        ksec = 0.5 * ksec + 0.5 * ksec_new

    # Post-processing from beam relations in the manual
    theta = np.zeros_like(y)
    M = np.zeros_like(y)
    V = np.zeros_like(y)
    psoil = np.zeros_like(y)

    for i in range(1, m - 1):
        theta[i] = (y[i + 1] - y[i - 1]) / (2.0 * dz)
        M[i] = EI * (y[i - 1] - 2.0 * y[i] + y[i + 1]) / dz**2
    theta[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * dz)
    theta[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / (2.0 * dz)
    M[0] = EI * (y[0] - 2.0 * y[1] + y[2]) / dz**2
    M[-1] = EI * (y[-3] - 2.0 * y[-2] + y[-1]) / dz**2

    for i in range(2, m - 2):
        V[i] = EI * (-y[i - 2] + 2.0 * y[i - 1] - 2.0 * y[i + 1] + y[i + 2]) / (2.0 * dz**3) + pile.P * theta[i]
        layer = dict(get_layer_at_depth(layers, z[i]))
        layer["pile_diameter_ref"] = pile.D
        psoil[i] = compute_py_curve(layer, z[i], abs(y[i]))
    V[0] = pile.H
    V[1] = V[2]
    V[-1] = 0.0
    V[-2] = V[-3]
    psoil[0] = 0.0
    psoil[1] = psoil[2]
    psoil[-1] = psoil[-2]

    return {
        "z": z,
        "y": y,
        "theta": theta,
        "M": M,
        "V": V,
        "p": psoil,
        "EI": np.full_like(z, EI),
        "A": np.full_like(z, math.pi * pile.D**2 / 4.0),
        "I": np.full_like(z, I),
    }


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def plot_profile_with_layers(results: Dict[str, np.ndarray], layers: List[Dict], pile: PileInputs):
    fig, ax = plt.subplots(figsize=(6, 9))
    for layer in layers:
        zt = safe_float(layer["z_top"])
        zb = safe_float(layer["z_bot"])
        stype = int(safe_float(layer["soilType"], 0))
        color = "#f4d03f" if stype == 2 else "#85c1e9"
        ax.axhspan(zt, zb, color=color, alpha=0.35)
        ax.text(0.02 * pile.D, 0.5 * (zt + zb), layer.get("name", "Layer"), va="center")

    scale = 1.0
    ymax = np.max(np.abs(results["y"]))
    if ymax > 1e-12:
        scale = min(25.0, 0.25 * pile.L / ymax)

    x_def = results["y"] * scale
    ax.plot(np.zeros_like(results["z"]), results["z"], "k--", lw=1)
    ax.plot(x_def, results["z"], lw=2, label="Pile under full load")
    ax.invert_yaxis()
    ax.set_xlabel(f"Scaled lateral displacement (x{scale:.1f})")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Pile deflected shape with soil layers")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_response(depth: np.ndarray, values: np.ndarray, title: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(values, depth, lw=2)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth (m)")
    ax.grid(True, alpha=0.3)
    return fig


# ------------------------------------------------------------
# Default app values
# ------------------------------------------------------------
DEFAULT_LAYERS = [
    {
        "name": "Sand",
        "z_top": 0.0,
        "z_bot": 15.0,
        "soilType": 2,
        "phi_deg": 35.0,
        "gamma": 17000.0,
        "k": 20000000.0,
        "Cd": 0.1,
        "model": "api_sand",
    },
    {
        "name": "Clay 2",
        "z_top": 15.0,
        "z_bot": 30.0,
        "soilType": 1,
        "c": 25000.0,
        "eps50": 0.02,
        "gamma": 17000.0,
        "Cd": 0.1,
        "model": "default",
    },
]


# ------------------------------------------------------------
# Streamlit GUI
# ------------------------------------------------------------
st.set_page_config(layout="wide", page_title="LPILE-Style Lateral Pile")
st.title("LPILE-Style Lateral Pile Analysis")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Pile and Load Inputs")
    head_condition = st.selectbox("Head condition", ["free", "fixed"], index=0)
    base_condition = st.selectbox("Base condition", ["pinned", "fixed"], index=0)
    H = st.number_input("Head lateral load H (N)", value=1.0e4, format="%.3e")
    P = st.number_input("Head axial load P (N)", value=0.0, format="%.3e")
    M = st.number_input("Head moment M (N.m)", value=0.0, format="%.3e")
    L = st.number_input("Pile length (m)", value=30.0, min_value=0.1, step=0.5)
    n_ele = st.number_input("Number of beam elements", value=48, min_value=8, step=2)
    D = st.number_input("Pile diameter (m)", value=0.80, min_value=0.05, step=0.05)
    E = st.number_input("Pile Young's modulus E (Pa)", value=3.0e10, format="%.3e")

    st.subheader("Soil Layers JSON")
    soil_json = st.text_area(
        "Edit the soil profile as JSON",
        value=json.dumps(DEFAULT_LAYERS, indent=2),
        height=320,
        label_visibility="collapsed",
    )

    run = st.button("Run analysis", type="primary")

with col2:
    st.info(
        "This version follows the manual-style workflow: nonlinear p-y curves plus finite-difference beam-column solution. "
        "Displacement, rotation, bending moment, shear, and soil reaction are recovered from the beam relations."
    )

if run:
    try:
        layers = json.loads(soil_json)
        if not isinstance(layers, list):
            raise ValueError("Soil layers JSON must be a list of layer objects.")

        pile = PileInputs(
            head_condition=head_condition,
            base_condition=base_condition,
            H=H,
            P=P,
            M=M,
            L=L,
            n_ele=int(n_ele),
            D=D,
            E=E,
        )

        errs = validate_layers(layers, pile.L)
        if errs:
            for e in errs:
                st.error(e)
            st.stop()

        table = build_layer_table(layers)
        st.subheader("Layer summary")
        st.dataframe(table, use_container_width=True)

        results = solve_beam_fd(pile, layers)

        top_disp = float(results["y"][0])
        max_m = float(np.max(np.abs(results["M"])))
        max_v = float(np.max(np.abs(results["V"])))

        c1, c2, c3 = st.columns(3)
        c1.metric("Top displacement", f"{top_disp:.6f} m")
        c2.metric("Max moment", f"{max_m/1e3:.3f} kN.m")
        c3.metric("Max shear", f"{max_v/1e3:.3f} kN")

        fig_profile = plot_profile_with_layers(results, layers, pile)
        st.pyplot(fig_profile, use_container_width=False)

        fig_y = plot_response(results["z"], results["y"], "Displacement (m)", "Lateral displacement (m)")
        st.pyplot(fig_y, use_container_width=False)

        fig_m = plot_response(results["z"], results["M"] / 1e3, "Bending Moment", "Moment (kN.m)")
        st.pyplot(fig_m, use_container_width=False)

        fig_v = plot_response(results["z"], results["V"] / 1e3, "Shear Force", "Shear (kN)")
        st.pyplot(fig_v, use_container_width=False)

        fig_p = plot_response(results["z"], results["p"] / 1e3, "Soil Reaction", "p (kN/m)")
        st.pyplot(fig_p, use_container_width=False)

        df_out = pd.DataFrame(
            {
                "Depth_m": results["z"],
                "Disp_m": results["y"],
                "Rotation_rad": results["theta"],
                "Moment_Nm": results["M"],
                "Shear_N": results["V"],
                "SoilReaction_Npm": results["p"],
            }
        )
        st.subheader("Results table")
        st.dataframe(df_out, use_container_width=True)

    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
    except np.linalg.LinAlgError:
        st.error("The finite-difference system became singular. Try more elements, different boundary conditions, or milder loads.")
    except Exception as e:
        st.exception(e)
