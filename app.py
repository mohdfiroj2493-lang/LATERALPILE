import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
except Exception:
    OPENSEES_AVAILABLE = False


st.set_page_config(page_title="Lateral Pile Analysis", layout="wide")
st.title("Lateral Pile Analysis with OpenSeesPy")


DEFAULT_LAYERS = [
    {
        "name": "Sand",
        "z_top": 0.0,
        "z_bot": 20.0,
        "soilType": 2,
        "gamma": 17000.0,
        "Cd": 0.1,
        "phi_deg": 40.0,
        "k": 2.0e7,
    }
]


if "soil_layers" not in st.session_state:
    st.session_state.soil_layers = json.loads(json.dumps(DEFAULT_LAYERS))


# ============================================================
# Helper functions
# ============================================================
def depth_modifier_pu(z: float, layer: Dict[str, Any]) -> float:
    return 1.0 + 0.03 * z


def depth_modifier_y50(z: float, layer: Dict[str, Any]) -> float:
    return 1.0


def get_layer(z: float, soil_layers: List[Dict[str, Any]]) -> Dict[str, Any]:
    for layer in soil_layers:
        if layer["z_top"] <= z < layer["z_bot"]:
            return layer
    if soil_layers and np.isclose(z, soil_layers[-1]["z_bot"]):
        return soil_layers[-1]
    raise ValueError(f"No soil layer defined for depth z = {z:.3f} m")


def tributary_length(node_index: int, n_nodes: int, dz: float) -> float:
    if node_index == 1 or node_index == n_nodes:
        return 0.5 * dz
    return dz


def derive_py_params(layer: Dict[str, Any], z: float, pile_diameter: float, tributary_len: float) -> Tuple[float, float]:
    soil_type = layer["soilType"]

    if soil_type == 1:
        if "c" not in layer:
            raise ValueError("Clay layer requires 'c'")
        c = float(layer["c"])
        eps50 = float(layer.get("eps50", 0.02))
        b = pile_diameter
        pu_per_m = max(1.0, 9.0 * c * b)
        y50 = max(1.0e-6, 2.5 * eps50 * b)

    elif soil_type == 2:
        if "phi_deg" not in layer:
            raise ValueError("Sand layer requires 'phi_deg'")
        if "gamma" not in layer:
            raise ValueError("Sand layer requires 'gamma'")

        phi_deg = float(layer["phi_deg"])
        gamma = float(layer["gamma"])
        gwtSwitch = int(layer.get("gwtSwitch", 1))
        kSwitch = int(layer.get("kSwitch", 1))
        puSwitch = int(layer.get("puSwitch", 1))

        b = pile_diameter
        z_eff = max(z, 1.0e-6)
        phi = np.radians(phi_deg)
        zb_ratio = z_eff / b
        pi = np.pi

        zb_vals = np.array([
            0.0000,0.1250,0.2500,0.3750,0.5000,0.6250,0.7500,0.8750,1.0000,1.1250,
            1.2500,1.3750,1.5000,1.6250,1.7500,1.8750,2.0000,2.1250,2.2500,2.3750,
            2.5000,2.6250,2.7500,2.8750,3.0000,3.1250,3.2500,3.3750,3.5000,3.6250,
            3.7500,3.8750,4.0000,4.1250,4.2500,4.3750,4.5000,4.6250,4.7500,4.8750,5.0000
        ])
        A_vals = np.array([
            2.8460,2.7105,2.6242,2.5257,2.4271,2.3409,2.2546,2.1437,2.0575,1.9589,
            1.8973,1.8111,1.7372,1.6632,1.5893,1.5277,1.4415,1.3799,1.3368,1.2690,
            1.2074,1.1581,1.1211,1.0780,1.0349,1.0164,0.9979,0.9733,0.9610,0.9487,
            0.9363,0.9117,0.8994,0.8994,0.8871,0.8871,0.8809,0.8809,0.8809,0.8809,0.8809
        ])
        A = 0.88 if zb_ratio >= 5.0 else np.interp(zb_ratio, zb_vals, A_vals)

        if puSwitch == 1:
            alpha = phi / 2.0
            beta = pi / 4.0 + phi / 2.0
            K0 = 0.4
            Ka = np.tan(pi / 4.0 - phi / 2.0) ** 2

            c1 = K0 * np.tan(phi) * np.sin(beta) / (np.tan(beta - phi) * np.cos(alpha))
            c2 = (np.tan(beta) / np.tan(beta - phi)) * np.tan(beta) * np.tan(alpha)
            c3 = K0 * np.tan(beta) * (np.tan(phi) * np.sin(beta) - np.tan(alpha))
            c4 = np.tan(beta) / np.tan(beta - phi) - Ka
            c5 = Ka * (np.tan(beta) ** 8 - 1.0)
            c6 = K0 * np.tan(phi) * (np.tan(beta) ** 4)

            pst = gamma * z_eff * (z_eff * (c1 + c2 + c3) + b * c4)
            psd = b * gamma * z_eff * (c5 + c6)
            pu_per_m = A * min(pst, psd)
        else:
            Kqo = np.exp((pi/2.0 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(pi/4.0 + phi/2.0) - np.exp(-(pi/2.0 - phi) * np.tan(phi)) * np.cos(phi) * np.tan(pi/4.0 - phi/2.0)
            Kco = (1.0/np.tan(phi)) * (np.exp((pi/2.0 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(pi/4.0 + phi/2.0) - 1.0)
            dcinf = 1.58 + 4.09 * (np.tan(phi) ** 4)
            Nc = (1.0/np.tan(phi)) * np.exp(pi * np.tan(phi)) * ((np.tan(pi/4.0 + phi/2.0) ** 2) - 1.0)
            Ko = 1.0 - np.sin(phi)
            Kcinf = Nc * dcinf
            Kqinf = Kcinf * Ko * np.tan(phi)
            aq = (Kqo / max(Kqinf - Kqo, 1.0e-9)) * (Ko * np.sin(phi) / np.sin(pi/4.0 + phi/2.0))
            KqD = (Kqo + Kqinf * aq * zb_ratio) / (1.0 + aq * zb_ratio)
            pu_per_m = gamma * z_eff * KqD * b

        pu_per_m = max(1.0, pu_per_m)

        ph_vals = np.array([28.8,29.5,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0])
        if gwtSwitch == 1:
            k_vals = np.array([10,23,45,61,80,100,120,140,160,182,215,250,275], dtype=float)
        else:
            k_vals = np.array([10,20,33,42,50,60,70,85,95,107,122,141,155], dtype=float)

        khat = np.interp(phi_deg, ph_vals, k_vals)
        k_si = khat * 271.45

        sig_v = max(z_eff * gamma, 0.01)
        if kSwitch == 2:
            c_sigma = np.sqrt(50.0 / sig_v)
            k_si = c_sigma * k_si

        atanh_half = 0.5 * np.log((1.0 + 0.5) / (1.0 - 0.5))
        y50 = 0.5 * (pu_per_m / max(A, 1.0e-9)) / (k_si * z_eff) * atanh_half
        y50 = max(1.0e-6, y50)

    else:
        raise ValueError("soilType must be 1 (clay) or 2 (sand)")

    pu_per_m *= depth_modifier_pu(z, layer)
    y50 *= depth_modifier_y50(z, layer)
    pult = pu_per_m * tributary_len
    return pult, y50


def validate_layers(layers: List[Dict[str, Any]], pile_length: float) -> List[str]:
    errors: List[str] = []
    if not layers:
        return ["At least one soil layer is required."]

    sorted_layers = sorted(layers, key=lambda x: float(x["z_top"]))

    if not np.isclose(float(sorted_layers[0]["z_top"]), 0.0):
        errors.append("The first layer must start at z_top = 0.0 m.")

    for i, layer in enumerate(sorted_layers):
        z_top = float(layer["z_top"])
        z_bot = float(layer["z_bot"])
        if z_bot <= z_top:
            errors.append(f"Layer '{layer['name']}' must have z_bot > z_top.")
        if int(layer["soilType"]) == 1 and "c" not in layer:
            errors.append(f"Clay layer '{layer['name']}' requires c.")
        if int(layer["soilType"]) == 2:
            if "phi_deg" not in layer:
                errors.append(f"Sand layer '{layer['name']}' requires phi_deg.")
            if "gamma" not in layer:
                errors.append(f"Sand layer '{layer['name']}' requires gamma.")
        if i > 0:
            prev = sorted_layers[i - 1]
            if not np.isclose(float(prev["z_bot"]), z_top):
                errors.append(f"Layers '{prev['name']}' and '{layer['name']}' must be continuous.")

    if not np.isclose(float(sorted_layers[-1]["z_bot"]), pile_length):
        errors.append("The last layer bottom depth must match the pile length.")

    return errors


# ============================================================
# OpenSees model functions
# ============================================================
def build_model(params: Dict[str, Any], soil_layers: List[Dict[str, Any]]):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    pile_length = params["PILE_LENGTH"]
    n_ele = params["N_ELE"]
    pile_diameter = params["PILE_DIAMETER"]
    e_pile = params["E_PILE"]
    pysimple1_c = params["PYSIMPLE1_C"]
    cd_default = params["CD_DEFAULT"]
    head_condition = params["HEAD_CONDITION"].lower()
    base_condition = params["BASE_CONDITION"].lower()

    n_node = n_ele + 1
    dz = pile_length / n_ele
    A = math.pi * pile_diameter**2 / 4.0
    I = math.pi * pile_diameter**4 / 64.0

    for i in range(n_node):
        node_tag = i + 1
        y = -i * dz
        ops.node(node_tag, 0.0, y)

    bottom = n_node
    if base_condition == "fixed":
        ops.fix(bottom, 1, 1, 1)
    elif base_condition == "pinned":
        ops.fix(bottom, 1, 1, 0)
    else:
        raise ValueError("BASE_CONDITION must be 'fixed' or 'pinned'")

    if head_condition == "fixed":
        ops.fix(1, 0, 0, 1)
    elif head_condition == "free":
        ops.fix(1, 0, 0, 0)
    else:
        raise ValueError("HEAD_CONDITION must be 'free' or 'fixed'")

    ops.geomTransf("Linear", 1)
    for e in range(1, n_node):
        ops.element("elasticBeamColumn", e, e, e + 1, A, e_pile, I, 1)

    spring_info = []
    mat_tag = 1000
    ele_tag = 2000
    soil_node_offset = 10000

    for pile_node in range(1, n_node):
        z = (pile_node - 1) * dz
        layer = get_layer(z, soil_layers)
        tlen = tributary_length(pile_node, n_node, dz)
        pult, y50 = derive_py_params(layer, z, pile_diameter, tlen)
        soil_type = int(layer["soilType"])
        Cd = float(layer.get("Cd", cd_default))

        soil_node = soil_node_offset + pile_node
        ops.node(soil_node, 0.0, -z)
        ops.fix(soil_node, 1, 1, 1)

        mat_tag += 1
        ops.uniaxialMaterial("PySimple1", mat_tag, soil_type, pult, y50, Cd, pysimple1_c)

        ele_tag += 1
        ops.element("zeroLength", ele_tag, pile_node, soil_node, "-mat", mat_tag, "-dir", 1)

        spring_info.append({
            "pile_node": pile_node,
            "spring_ele": ele_tag,
            "z": z,
            "layer": layer["name"],
            "soilType": soil_type,
            "pult": pult,
            "y50": y50,
        })

    return spring_info


def run_static_lateral_analysis(params: Dict[str, Any]) -> int:
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(1, params["HEAD_LATERAL_LOAD"], -params["HEAD_AXIAL_LOAD"], params["HEAD_MOMENT"])

    ops.constraints("Plain")
    ops.numberer("Plain")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1.0e-8, 100)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.05)
    ops.analysis("Static")

    ok = ops.analyze(20)
    if ok != 0:
        ops.algorithm("ModifiedNewton")
        ok = ops.analyze(40)
    return ok


def get_results(n_ele: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_node = n_ele + 1
    depth, ux, uy, rz = [], [], [], []
    for n in range(1, n_node + 1):
        _, y = ops.nodeCoord(n)
        d = ops.nodeDisp(n)
        depth.append(-y)
        ux.append(d[0])
        uy.append(d[1])
        rz.append(d[2])
    return np.array(depth), np.array(ux), np.array(uy), np.array(rz)


def get_spring_reactions(pile_length: float, n_ele: int, spring_sign: float) -> Tuple[np.ndarray, np.ndarray]:
    n_node = n_ele + 1
    dz = pile_length / n_ele
    z_spring, p_spring = [], []
    first_spring_ele = 2001

    for i in range(1, n_node):
        z = (i - 1) * dz
        spring_ele = first_spring_ele + (i - 1)
        sf = ops.eleForce(spring_ele)
        p = spring_sign * sf[0]
        z_spring.append(z)
        p_spring.append(p)

    z_spring = np.array(z_spring)
    p_spring = np.array(p_spring)
    if np.sum(p_spring) < 0.0:
        p_spring = -p_spring
    return z_spring, p_spring


def get_beam_force_profiles(pile_length: float, n_ele: int) -> Dict[str, np.ndarray]:
    dz = pile_length / n_ele
    z_top, z_bot = [], []
    V_top, V_bot = [], []
    M_top, M_bot = [], []
    P_top, P_bot = [], []

    for e in range(1, n_ele + 1):
        f = ops.eleForce(e)
        zt = (e - 1) * dz
        zb = e * dz
        m_top = f[2]
        m_bot = -f[5]
        v_ele = -(m_bot - m_top) / dz

        z_top.append(zt)
        z_bot.append(zb)
        P_top.append(f[0])
        P_bot.append(-f[3])
        V_top.append(v_ele)
        V_bot.append(v_ele)
        M_top.append(m_top)
        M_bot.append(m_bot)

    return {
        "z_top": np.array(z_top),
        "z_bot": np.array(z_bot),
        "P_top": np.array(P_top),
        "P_bot": np.array(P_bot),
        "V_top": np.array(V_top),
        "V_bot": np.array(V_bot),
        "M_top": np.array(M_top),
        "M_bot": np.array(M_bot),
    }


def build_node_force_arrays(force_data: Dict[str, np.ndarray], pile_length: float, n_ele: int):
    n_node = n_ele + 1
    z_node = np.linspace(0.0, pile_length, n_node)
    V_node = np.zeros(n_node)
    M_node = np.zeros(n_node)
    V_node[0] = force_data["V_top"][0]
    M_node[0] = force_data["M_top"][0]
    for i in range(1, n_node):
        V_node[i] = force_data["V_bot"][i - 1]
        M_node[i] = force_data["M_bot"][i - 1]
    return z_node, V_node, M_node


def plot_deformed_pile(depth, ux, soil_layers, title, scale=None):
    fig, ax = plt.subplots(figsize=(7, 9))

    colors = ["#f3e5ab", "#d7ecff", "#d9f2d9", "#f8d7da", "#e7d4f5", "#ffe6cc"]
    for i, layer in enumerate(soil_layers):
        c = colors[i % len(colors)]
        ax.axhspan(layer["z_top"], layer["z_bot"], color=c, alpha=0.6)
        ax.text(0.02, 0.5 * (layer["z_top"] + layer["z_bot"]), layer["name"],
                transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=9)

    max_abs = max(float(np.max(np.abs(ux))), 1e-9)
    if scale is None:
        scale = max(1.0, 0.15 * float(np.max(depth)) / max_abs)

    x_undeformed = np.zeros_like(depth)
    x_deformed = ux * scale

    ax.plot(x_undeformed, depth, "k--", linewidth=1.5, label="Undeformed pile")
    ax.plot(x_deformed, depth, "b-", linewidth=2.5, label=f"Deformed pile (x{scale:.1f})")

    ax.invert_yaxis()
    ax.set_xlabel("Horizontal position (scaled)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_profile(x, y, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(x, y, marker="o")
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_moment_profile(force_data, title):
    fig, ax = plt.subplots(figsize=(6, 8))
    z_plot = []
    m_plot = []
    for i in range(len(force_data["z_top"])):
        z_plot.extend([force_data["z_top"][i], force_data["z_bot"][i]])
        m_plot.extend([force_data["M_top"][i], force_data["M_bot"][i]])
    ax.plot(np.array(m_plot) / 1e3, z_plot, marker="o")
    ax.invert_yaxis()
    ax.set_xlabel("Bending moment M (kN.m)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ============================================================
# Sidebar inputs
# ============================================================
st.sidebar.header("Pile and Load Inputs")
HEAD_CONDITION = st.sidebar.selectbox("Head condition", ["free", "fixed"], index=0)
BASE_CONDITION = st.sidebar.selectbox("Base condition", ["pinned", "fixed"], index=0)
HEAD_LATERAL_LOAD = st.sidebar.number_input("Head lateral load H (N)", value=2.0e5, step=1.0e4, format="%.3e")
HEAD_AXIAL_LOAD = st.sidebar.number_input("Head axial load P (N)", value=0.0, step=1.0e4, format="%.3e")
HEAD_MOMENT = st.sidebar.number_input("Head moment M (N.m)", value=0.0, step=1.0e4, format="%.3e")
PILE_LENGTH = st.sidebar.number_input("Pile length (m)", min_value=1.0, value=20.0, step=1.0)
N_ELE = st.sidebar.number_input("Number of beam elements", min_value=4, value=48, step=1)
PILE_DIAMETER = st.sidebar.number_input("Pile diameter (m)", min_value=0.05, value=0.8, step=0.05)
E_PILE = st.sidebar.number_input("Pile Young's modulus E (Pa)", min_value=1.0e6, value=30.0e9, step=1.0e9, format="%.3e")
PYSIMPLE1_C = st.sidebar.number_input("PySimple1 C", value=0.0, step=0.1)
CD_DEFAULT = st.sidebar.number_input("Default Cd", value=0.1, step=0.05)
SPRING_SIGN = st.sidebar.number_input("Spring sign", value=1.0, step=1.0)


# ============================================================
# Soil layer GUI
# ============================================================
st.subheader("Soil Layers")

col_add1, col_add2, col_add3 = st.columns([1, 1, 4])
with col_add1:
    if st.button("Add sand layer"):
        next_top = st.session_state.soil_layers[-1]["z_bot"] if st.session_state.soil_layers else 0.0
        st.session_state.soil_layers.append({
            "name": f"Sand {len(st.session_state.soil_layers) + 1}",
            "z_top": float(next_top),
            "z_bot": float(next_top) + 2.0,
            "soilType": 2,
            "gamma": 17000.0,
            "Cd": 0.1,
            "phi_deg": 35.0,
            "k": 2.0e7,
        })
with col_add2:
    if st.button("Add clay layer"):
        next_top = st.session_state.soil_layers[-1]["z_bot"] if st.session_state.soil_layers else 0.0
        st.session_state.soil_layers.append({
            "name": f"Clay {len(st.session_state.soil_layers) + 1}",
            "z_top": float(next_top),
            "z_bot": float(next_top) + 2.0,
            "soilType": 1,
            "gamma": 17000.0,
            "Cd": 0.1,
            "c": 25000.0,
            "eps50": 0.02,
        })

updated_layers = []
for i, layer in enumerate(st.session_state.soil_layers):
    with st.expander(f"Layer {i + 1}: {layer.get('name', 'Layer')}", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        name = c1.text_input("Name", value=str(layer.get("name", f"Layer {i + 1}")), key=f"name_{i}")
        soil_type_label = c2.selectbox("Soil type", ["Sand", "Clay"], index=0 if int(layer.get("soilType", 2)) == 2 else 1, key=f"soil_type_{i}")
        z_top = c3.number_input("z_top (m)", value=float(layer.get("z_top", 0.0)), step=0.5, key=f"z_top_{i}")
        z_bot = c4.number_input("z_bot (m)", value=float(layer.get("z_bot", 1.0)), step=0.5, key=f"z_bot_{i}")

        c5, c6 = st.columns(2)
        gamma = c5.number_input("gamma (N/m3)", value=float(layer.get("gamma", 17000.0)), step=500.0, key=f"gamma_{i}")
        cd = c6.number_input("Cd", value=float(layer.get("Cd", 0.1)), step=0.05, key=f"Cd_{i}")

        new_layer = {
            "name": name,
            "z_top": float(z_top),
            "z_bot": float(z_bot),
            "gamma": float(gamma),
            "Cd": float(cd),
        }

        if soil_type_label == "Sand":
            c7, c8 = st.columns(2)
            phi_deg = c7.number_input("phi_deg (deg)", value=float(layer.get("phi_deg", 35.0)), step=1.0, key=f"phi_{i}")
            k_val = c8.number_input("k (N/m3)", value=float(layer.get("k", 2.0e7)), step=1.0e6, format="%.3e", key=f"k_{i}")
            new_layer.update({
                "soilType": 2,
                "phi_deg": float(phi_deg),
                "k": float(k_val),
            })
        else:
            c7, c8 = st.columns(2)
            cohesion = c7.number_input("c (Pa)", value=float(layer.get("c", 25000.0)), step=1000.0, key=f"c_{i}")
            eps50 = c8.number_input("eps50", value=float(layer.get("eps50", 0.02)), step=0.005, format="%.4f", key=f"eps50_{i}")
            new_layer.update({
                "soilType": 1,
                "c": float(cohesion),
                "eps50": float(eps50),
            })

        if st.button(f"Delete layer {i + 1}", key=f"delete_{i}"):
            continue

        updated_layers.append(new_layer)

st.session_state.soil_layers = updated_layers
soil_layers = sorted(st.session_state.soil_layers, key=lambda x: x["z_top"])

params = {
    "HEAD_CONDITION": HEAD_CONDITION,
    "BASE_CONDITION": BASE_CONDITION,
    "HEAD_LATERAL_LOAD": HEAD_LATERAL_LOAD,
    "HEAD_AXIAL_LOAD": HEAD_AXIAL_LOAD,
    "HEAD_MOMENT": HEAD_MOMENT,
    "PILE_LENGTH": PILE_LENGTH,
    "N_ELE": int(N_ELE),
    "PILE_DIAMETER": PILE_DIAMETER,
    "E_PILE": E_PILE,
    "PYSIMPLE1_C": PYSIMPLE1_C,
    "CD_DEFAULT": CD_DEFAULT,
    "SPRING_SIGN": SPRING_SIGN,
}

errors = validate_layers(soil_layers, PILE_LENGTH)
if errors:
    for err in errors:
        st.error(err)

st.dataframe(pd.DataFrame(soil_layers), use_container_width=True)

run_clicked = st.button("Run analysis", type="primary", disabled=(len(errors) > 0))

if not OPENSEES_AVAILABLE:
    st.warning("OpenSeesPy is not installed in this environment. Install it to run the analysis.")

if run_clicked and OPENSEES_AVAILABLE and not errors:
    try:
        spring_info = build_model(params, soil_layers)
        ok = run_static_lateral_analysis(params)

        if ok != 0:
            st.error("Analysis did not converge.")
        else:
            depth, ux, uy, rz = get_results(int(N_ELE))
            z_spring, p_spring = get_spring_reactions(PILE_LENGTH, int(N_ELE), SPRING_SIGN)
            force_data = get_beam_force_profiles(PILE_LENGTH, int(N_ELE))
            z_node, V_node, M_node = build_node_force_arrays(force_data, PILE_LENGTH, int(N_ELE))

            c1, c2, c3 = st.columns(3)
            c1.metric("Head disp Ux (mm)", f"{ux[0] * 1000:.3f}")
            c2.metric("Max |V| (kN)", f"{np.max(np.abs(V_node)) / 1e3:.3f}")
            c3.metric("Max |M| (kN.m)", f"{np.max(np.abs(M_node)) / 1e3:.3f}")

            st.pyplot(plot_deformed_pile(depth, ux, soil_layers, f"Pile deformation in soil layers ({HEAD_CONDITION}-head, {BASE_CONDITION}-base)"))

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.pyplot(plot_profile(ux * 1000.0, depth, "Lateral deflection (mm)", "Depth below ground surface (m)", "Deflection profile"))
                st.pyplot(plot_profile(p_spring / 1e3, z_spring, "Soil reaction p (kN)", "Depth below ground surface (m)", "Soil reaction profile"))
            with pcol2:
                st.pyplot(plot_profile(force_data["V_top"] / 1e3, force_data["z_top"], "Shear force V (kN)", "Depth below ground surface (m)", "Shear profile"))
                st.pyplot(plot_moment_profile(force_data, "Bending moment profile"))

            spring_df = pd.DataFrame(spring_info)
            node_df = pd.DataFrame({
                "Node": np.arange(1, len(z_node) + 1),
                "z_m": z_node,
                "V_kN": V_node / 1e3,
                "M_kNm": M_node / 1e3,
                "Ux_m": ux,
                "Uy_m": uy,
                "Rz_rad": rz,
            })
            ele_df = pd.DataFrame({
                "Element": np.arange(1, len(force_data["z_top"]) + 1),
                "zTop_m": force_data["z_top"],
                "zBot_m": force_data["z_bot"],
                "Vtop_kN": force_data["V_top"] / 1e3,
                "Vbot_kN": force_data["V_bot"] / 1e3,
                "Mtop_kNm": force_data["M_top"] / 1e3,
                "Mbot_kNm": force_data["M_bot"] / 1e3,
            })

            st.subheader("Spring Properties")
            st.dataframe(spring_df, use_container_width=True)

            st.subheader("Node Force Table")
            st.dataframe(node_df, use_container_width=True)
            st.download_button("Download node results CSV", node_df.to_csv(index=False).encode("utf-8"), file_name="node_results.csv", mime="text/csv")

            st.subheader("Element End Force Table")
            st.dataframe(ele_df, use_container_width=True)
            st.download_button("Download element results CSV", ele_df.to_csv(index=False).encode("utf-8"), file_name="element_results.csv", mime="text/csv")

    except Exception as exc:
        st.exception(exc)
