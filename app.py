import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import openseespy.opensees as ops
except Exception as e:
    ops = None
    OPENSEES_IMPORT_ERROR = str(e)
else:
    OPENSEES_IMPORT_ERROR = None


st.set_page_config(page_title="Pile Lateral Analysis", layout="wide")


# ============================================================
# DEFAULTS
# ============================================================
DEFAULT_SOIL_LAYERS = [
    {
        "name": "Loose sand",
        "z_top": 0.0,
        "z_bot": 6.0,
        "soilType": 2,
        "phi_deg": 30.0,
        "gamma": 17000.0,
        "k": 20000000.0,
        "Cd": 0.1,
    },
    {
        "name": "Soft clay",
        "z_top": 6.0,
        "z_bot": 14.0,
        "soilType": 1,
        "c": 25000.0,
        "eps50": 0.02,
        "gamma": 16500.0,
        "Cd": 0.1,
    },
    {
        "name": "Dense sand",
        "z_top": 14.0,
        "z_bot": 24.0,
        "soilType": 2,
        "phi_deg": 38.0,
        "gamma": 19000.0,
        "k": 40000000.0,
        "Cd": 0.1,
    },
]

LAYER_COLORS = {
    1: "#b07d62",  # clay
    2: "#d9c27a",  # sand
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def depth_modifier_pu(z, layer):
    return 1.0 + 0.03 * z


def depth_modifier_y50(z, layer):
    return 1.0


def tributary_length(node_index, n_nodes, dz):
    if node_index == 1 or node_index == n_nodes:
        return 0.5 * dz
    return dz


def get_layer(z, soil_layers):
    for layer in soil_layers:
        if layer["z_top"] <= z < layer["z_bot"]:
            return layer
    if np.isclose(z, soil_layers[-1]["z_bot"]):
        return soil_layers[-1]
    raise ValueError(f"No soil layer defined for depth z = {z:.3f} m")


def validate_soil_layers(soil_layers, pile_length):
    if not soil_layers:
        raise ValueError("At least one soil layer is required.")

    sorted_layers = sorted(soil_layers, key=lambda x: x["z_top"])

    if abs(sorted_layers[0]["z_top"] - 0.0) > 1e-9:
        raise ValueError("The first soil layer must start at z_top = 0.0 m.")

    prev_bot = None
    for i, layer in enumerate(sorted_layers, start=1):
        if layer["z_bot"] <= layer["z_top"]:
            raise ValueError(f"Layer {i} has z_bot <= z_top.")
        if prev_bot is not None and abs(layer["z_top"] - prev_bot) > 1e-6:
            raise ValueError("Soil layers must be continuous with no gap or overlap.")
        if layer["soilType"] == 1 and "c" not in layer:
            raise ValueError(f"Clay layer '{layer['name']}' requires c.")
        if layer["soilType"] == 2 and "phi_deg" not in layer:
            raise ValueError(f"Sand layer '{layer['name']}' requires phi_deg.")
        if "gamma" not in layer:
            raise ValueError(f"Layer '{layer['name']}' requires gamma.")
        prev_bot = layer["z_bot"]

    if abs(sorted_layers[-1]["z_bot"] - pile_length) > 1e-6:
        raise ValueError("The last soil layer z_bot must match the pile length.")

    return sorted_layers


def derive_py_params(layer, z, pile_diameter, tributary_len):
    soil_type = layer["soilType"]

    if soil_type == 1:
        if "c" not in layer:
            raise ValueError("Clay layer requires 'c'")

        c = layer["c"]
        eps50 = layer.get("eps50", 0.02)
        b = pile_diameter

        pu_per_m = max(1.0, 9.0 * c * b)
        y50 = max(1.0e-6, 2.5 * eps50 * b)

    elif soil_type == 2:
        if "phi_deg" not in layer:
            raise ValueError("Sand layer requires 'phi_deg'")
        if "gamma" not in layer:
            raise ValueError("Sand layer requires 'gamma'")

        phi_deg = layer["phi_deg"]
        gamma = layer["gamma"]
        gwtSwitch = layer.get("gwtSwitch", 1)
        kSwitch = layer.get("kSwitch", 1)
        puSwitch = layer.get("puSwitch", 1)

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
            Kqo = np.exp((pi/2.0 + phi) * np.tan(phi)) * np.cos(phi) * np.tan(pi/4.0 + phi/2.0) \
                - np.exp(-(pi/2.0 - phi) * np.tan(phi)) * np.cos(phi) * np.tan(pi/4.0 - phi/2.0)
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
        raise ValueError("soilType must be 1 (c-type clay) or 2 (API/Reese sand)")

    pu_per_m *= depth_modifier_pu(z, layer)
    y50 *= depth_modifier_y50(z, layer)

    pult = pu_per_m * tributary_len
    return pult, y50


# ============================================================
# OPENSees MODEL AND ANALYSIS
# ============================================================
def build_model(config, soil_layers):
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)

    n_ele = config["n_ele"]
    pile_length = config["pile_length"]
    pile_diameter = config["pile_diameter"]
    e_pile = config["e_pile"]
    head_condition = config["head_condition"]
    base_condition = config["base_condition"]
    pysimple1_c = config["pysimple1_c"]
    cd_default = config["cd_default"]

    n_node = n_ele + 1
    dz = pile_length / n_ele

    area = np.pi * pile_diameter**2 / 4.0
    inertia = np.pi * pile_diameter**4 / 64.0

    for i in range(n_node):
        node_tag = i + 1
        y = -i * dz
        ops.node(node_tag, 0.0, y)

    bottom = n_node
    if base_condition.lower() == "fixed":
        ops.fix(bottom, 1, 1, 1)
    elif base_condition.lower() == "pinned":
        ops.fix(bottom, 1, 1, 0)
    else:
        raise ValueError("Base condition must be 'fixed' or 'pinned'")

    if head_condition.lower() == "fixed":
        ops.fix(1, 0, 0, 1)
    elif head_condition.lower() == "free":
        ops.fix(1, 0, 0, 0)
    else:
        raise ValueError("Head condition must be 'free' or 'fixed'")

    ops.geomTransf('Linear', 1)
    for e in range(1, n_node):
        ops.element('elasticBeamColumn', e, e, e + 1, area, e_pile, inertia, 1)

    spring_info = []
    mat_tag = 1000
    ele_tag = 2000
    soil_node_offset = 10000

    for pile_node in range(1, n_node):
        z = (pile_node - 1) * dz
        layer = get_layer(z, soil_layers)
        tlen = tributary_length(pile_node, n_node, dz)

        pult, y50 = derive_py_params(layer, z, pile_diameter, tlen)

        soil_type = layer["soilType"]
        cd = layer.get("Cd", cd_default)

        soil_node = soil_node_offset + pile_node
        ops.node(soil_node, 0.0, -z)
        ops.fix(soil_node, 1, 1, 1)

        mat_tag += 1
        ops.uniaxialMaterial('PySimple1', mat_tag, soil_type, pult, y50, cd, pysimple1_c)

        ele_tag += 1
        ops.element('zeroLength', ele_tag, pile_node, soil_node, '-mat', mat_tag, '-dir', 1)

        spring_info.append({
            "pile_node": pile_node,
            "spring_ele": ele_tag,
            "z_m": z,
            "layer": layer["name"],
            "soilType": soil_type,
            "pult_N": pult,
            "y50_m": y50,
            "input_mode": "c-type" if soil_type == 1 else "phi-type"
        })

    return spring_info


def run_static_lateral_analysis(config):
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(1, config["head_lateral_load"], -config["head_axial_load"], config["head_moment"])

    ops.constraints('Plain')
    ops.numberer('Plain')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-8, 100)
    ops.algorithm('Newton')
    ops.integrator('LoadControl', 0.05)
    ops.analysis('Static')

    ok = ops.analyze(20)
    if ok != 0:
        ops.algorithm('ModifiedNewton')
        ok = ops.analyze(40)
    return ok


def get_results(config):
    n_node = config["n_ele"] + 1
    depth, ux, uy, rz = [], [], [], []

    for n in range(1, n_node + 1):
        _, y = ops.nodeCoord(n)
        d = ops.nodeDisp(n)
        depth.append(-y)
        ux.append(d[0])
        uy.append(d[1])
        rz.append(d[2])

    return np.array(depth), np.array(ux), np.array(uy), np.array(rz)


def get_spring_reactions(config):
    n_node = config["n_ele"] + 1
    dz = config["pile_length"] / config["n_ele"]
    spring_sign = config["spring_sign"]

    z_spring = []
    p_spring = []
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


def get_beam_force_profiles(config):
    dz = config["pile_length"] / config["n_ele"]
    z_top, z_bot = [], []
    v_top, v_bot = [], []
    m_top, m_bot = [], []
    p_top, p_bot = [], []

    for e in range(1, config["n_ele"] + 1):
        f = ops.eleForce(e)
        zt = (e - 1) * dz
        zb = e * dz

        m_t = f[2]
        m_b = -f[5]
        v_ele = -(m_b - m_t) / dz

        z_top.append(zt)
        z_bot.append(zb)
        p_top.append(f[0])
        p_bot.append(-f[3])
        v_top.append(v_ele)
        v_bot.append(v_ele)
        m_top.append(m_t)
        m_bot.append(m_b)

    return {
        "z_top": np.array(z_top),
        "z_bot": np.array(z_bot),
        "P_top": np.array(p_top),
        "P_bot": np.array(p_bot),
        "V_top": np.array(v_top),
        "V_bot": np.array(v_bot),
        "M_top": np.array(m_top),
        "M_bot": np.array(m_bot),
    }


def build_node_force_arrays(config, force_data):
    n_node = config["n_ele"] + 1
    z_node = np.linspace(0.0, config["pile_length"], n_node)
    v_node = np.zeros(n_node)
    m_node = np.zeros(n_node)

    v_node[0] = force_data["V_top"][0]
    m_node[0] = force_data["M_top"][0]

    for i in range(1, n_node):
        v_node[i] = force_data["V_bot"][i - 1]
        m_node[i] = force_data["M_bot"][i - 1]

    return z_node, v_node, m_node


def run_analysis(config, soil_layers):
    spring_info = build_model(config, soil_layers)
    ok = run_static_lateral_analysis(config)
    if ok != 0:
        raise RuntimeError("Analysis did not converge.")

    depth, ux, uy, rz = get_results(config)
    z_spring, p_spring = get_spring_reactions(config)
    force_data = get_beam_force_profiles(config)
    z_node, v_node, m_node = build_node_force_arrays(config, force_data)

    spring_df = pd.DataFrame(spring_info)
    node_force_df = pd.DataFrame({
        "Node": np.arange(1, len(z_node) + 1),
        "z_m": z_node,
        "V_kN": v_node / 1e3,
        "M_kNm": m_node / 1e3,
    })
    element_df = pd.DataFrame({
        "Ele": np.arange(1, config["n_ele"] + 1),
        "zTop_m": force_data["z_top"],
        "zBot_m": force_data["z_bot"],
        "Vtop_kN": force_data["V_top"] / 1e3,
        "Vbot_kN": force_data["V_bot"] / 1e3,
        "Mtop_kNm": force_data["M_top"] / 1e3,
        "Mbot_kNm": force_data["M_bot"] / 1e3,
    })

    summary = {
        "head_disp_mm": ux[0] * 1000.0,
        "head_rot_rad": rz[0],
        "base_rot_rad": rz[-1],
        "max_soil_reaction_kN": np.max(np.abs(p_spring)) / 1e3,
        "max_shear_kN": np.max(np.abs(v_node)) / 1e3,
        "max_moment_kNm": np.max(np.abs(m_node)) / 1e3,
        "top_moment_kNm": m_node[0] / 1e3,
        "bottom_moment_kNm": m_node[-1] / 1e3,
    }

    return {
        "summary": summary,
        "depth": depth,
        "ux": ux,
        "uy": uy,
        "rz": rz,
        "z_spring": z_spring,
        "p_spring": p_spring,
        "force_data": force_data,
        "spring_df": spring_df,
        "node_force_df": node_force_df,
        "element_df": element_df,
    }


# ============================================================
# PLOTTING
# ============================================================
def plot_deflection(depth, ux, title):
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(ux * 1000.0, depth, marker='o')
    ax.invert_yaxis()
    ax.set_xlabel("Lateral deflection (mm)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_soil_reaction_profile(z_spring, p_spring, title):
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(p_spring / 1e3, z_spring, marker='o')
    ax.invert_yaxis()
    ax.set_xlabel("Soil reaction p (kN)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_shear_profile(force_data, title):
    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(force_data["V_top"] / 1e3, force_data["z_top"], marker='o')
    ax.invert_yaxis()
    ax.set_xlabel("Shear force V (kN)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_moment_profile(force_data, title):
    z_plot = []
    m_plot = []
    for i in range(len(force_data["z_top"])):
        z_plot.extend([force_data["z_top"][i], force_data["z_bot"][i]])
        m_plot.extend([force_data["M_top"][i], force_data["M_bot"][i]])

    fig, ax = plt.subplots(figsize=(5, 7))
    ax.plot(np.array(m_plot) / 1e3, z_plot, marker='o')
    ax.invert_yaxis()
    ax.set_xlabel("Bending moment M (kN.m)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_pile_soil_profile(depth, ux, soil_layers, pile_diameter, title):
    fig, ax = plt.subplots(figsize=(6, 8))

    max_disp = float(np.max(np.abs(ux))) if len(ux) else 0.0
    scale = 1.0
    if max_disp > 0:
        scale = min(25.0, max(3.0, 0.25 * max(layer["z_bot"] for layer in soil_layers) / max_disp))

    half_width = pile_diameter / 2.0
    undeformed_x = np.zeros_like(depth)
    deformed_x = ux * scale

    for layer in soil_layers:
        color = LAYER_COLORS.get(layer["soilType"], "#dddddd")
        label = f"{layer['name']} ({'Clay' if layer['soilType']==1 else 'Sand'})"
        ax.axhspan(layer["z_top"], layer["z_bot"], color=color, alpha=0.45, label=label)
        ax.text(
            1.15 * pile_diameter,
            0.5 * (layer["z_top"] + layer["z_bot"]),
            layer["name"],
            va="center",
            fontsize=9,
        )

    ax.plot(undeformed_x, depth, linestyle="--", linewidth=1.8, label="Pile centerline (undeformed)")
    ax.plot(deformed_x, depth, linewidth=2.4, label=f"Pile centerline (deformed x{scale:.1f})")
    ax.plot(np.full_like(depth, -half_width), depth, alpha=0.6, linewidth=1.0)
    ax.plot(np.full_like(depth, half_width), depth, alpha=0.6, linewidth=1.0)

    ax.scatter([deformed_x[0]], [depth[0]], marker=">", s=90, label="Loaded head")
    ax.axhline(0.0, linewidth=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Horizontal position / deformed shape (m)")
    ax.set_ylabel("Depth below ground surface (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_h = []
    unique_l = []
    for h, l in zip(handles, labels):
        if l not in seen:
            unique_h.append(h)
            unique_l.append(l)
            seen.add(l)
    ax.legend(unique_h, unique_l, loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================
# STREAMLIT UI
# ============================================================
st.title("Pile Lateral Analysis with OpenSeesPy + Streamlit")
st.caption("Nonlinear p-y springs using PySimple1, with GUI-based soil layers, force plots, and deformed pile profile.")

if ops is None:
    st.error(f"OpenSeesPy could not be imported: {OPENSEES_IMPORT_ERROR}")
    st.stop()

if "soil_layers" not in st.session_state:
    st.session_state.soil_layers = [dict(layer) for layer in DEFAULT_SOIL_LAYERS]

with st.sidebar:
    st.header("Model Inputs")

    head_condition = st.selectbox("Head condition", ["free", "fixed"], index=0)
    base_condition = st.selectbox("Base condition", ["pinned", "fixed"], index=0)

    st.subheader("Loads")
    head_lateral_load = st.number_input("Head lateral load H (N)", value=2.0e5, step=1.0e4, format="%.3e")
    head_axial_load = st.number_input("Head axial load P (N)", value=0.0, step=1.0e4, format="%.3e")
    head_moment = st.number_input("Head moment M (N.m)", value=0.0, step=1.0e4, format="%.3e")

    st.subheader("Pile")
    pile_length = st.number_input("Pile length (m)", value=24.0, min_value=0.1, step=0.5)
    n_ele = st.number_input("Number of elements", value=48, min_value=2, step=1)
    pile_diameter = st.number_input("Pile diameter (m)", value=0.8, min_value=0.01, step=0.05)
    e_pile = st.number_input("Pile modulus E (Pa)", value=30e9, step=1e9, format="%.3e")

    st.subheader("Spring Settings")
    pysimple1_c = st.number_input("PySimple1 c", value=0.0, step=0.1)
    cd_default = st.number_input("Default Cd", value=0.1, step=0.05)
    spring_sign = st.number_input("Spring sign", value=1.0, step=1.0)

st.subheader("Soil Layers")
st.caption("Edit soil layers directly in the GUI. Keep layers continuous from 0 m to the pile tip.")

soil_layers = st.session_state.soil_layers
soil_cols = st.columns([1, 1])
with soil_cols[0]:
    if st.button("Add sand layer"):
        last_bot = soil_layers[-1]["z_bot"] if soil_layers else 0.0
        soil_layers.append({
            "name": f"Sand {len(soil_layers)+1}",
            "z_top": last_bot,
            "z_bot": last_bot + 1.0,
            "soilType": 2,
            "phi_deg": 30.0,
            "gamma": 18000.0,
            "k": 20000000.0,
            "Cd": 0.1,
        })
with soil_cols[1]:
    if st.button("Add clay layer"):
        last_bot = soil_layers[-1]["z_bot"] if soil_layers else 0.0
        soil_layers.append({
            "name": f"Clay {len(soil_layers)+1}",
            "z_top": last_bot,
            "z_bot": last_bot + 1.0,
            "soilType": 1,
            "c": 25000.0,
            "eps50": 0.02,
            "gamma": 17000.0,
            "Cd": 0.1,
        })

remove_index = None
for i, layer in enumerate(soil_layers):
    with st.expander(f"Layer {i+1}: {layer.get('name', 'Layer')}", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            layer["name"] = st.text_input("Name", value=layer.get("name", f"Layer {i+1}"), key=f"name_{i}")
        with c2:
            layer_type_label = st.selectbox(
                "Type",
                ["Sand", "Clay"],
                index=0 if layer.get("soilType", 2) == 2 else 1,
                key=f"type_{i}"
            )
            layer["soilType"] = 2 if layer_type_label == "Sand" else 1
        with c3:
            if len(soil_layers) > 1 and st.button("Delete layer", key=f"del_{i}"):
                remove_index = i

        c4, c5, c6, c7 = st.columns(4)
        with c4:
            layer["z_top"] = st.number_input("z_top (m)", value=float(layer.get("z_top", 0.0)), step=0.5, key=f"zt_{i}")
        with c5:
            layer["z_bot"] = st.number_input("z_bot (m)", value=float(layer.get("z_bot", 1.0)), step=0.5, key=f"zb_{i}")
        with c6:
            layer["gamma"] = st.number_input("gamma (N/m3)", value=float(layer.get("gamma", 18000.0)), step=500.0, key=f"ga_{i}")
        with c7:
            layer["Cd"] = st.number_input("Cd", value=float(layer.get("Cd", 0.1)), step=0.05, key=f"cd_{i}")

        if layer["soilType"] == 2:
            s1, s2 = st.columns(2)
            with s1:
                layer["phi_deg"] = st.number_input("phi (deg)", value=float(layer.get("phi_deg", 30.0)), step=1.0, key=f"phi_{i}")
            with s2:
                layer["k"] = st.number_input("k (N/m3)", value=float(layer.get("k", 2.0e7)), step=1.0e6, format="%.3e", key=f"k_{i}")
            layer.pop("c", None)
            layer.pop("eps50", None)
        else:
            s1, s2 = st.columns(2)
            with s1:
                layer["c"] = st.number_input("c (Pa)", value=float(layer.get("c", 25000.0)), step=1000.0, key=f"c_{i}")
            with s2:
                layer["eps50"] = st.number_input("eps50", value=float(layer.get("eps50", 0.02)), step=0.005, format="%.3f", key=f"eps_{i}")
            layer.pop("phi_deg", None)
            layer.pop("k", None)

if remove_index is not None:
    soil_layers.pop(remove_index)
    st.session_state.soil_layers = soil_layers
    st.rerun()

st.code(json.dumps(soil_layers, indent=2), language="json")

run_btn = st.button("Run analysis", type="primary")

st.markdown(
    """
    **Layer rules**
    - `soilType = 1` for clay layers using `c` and optional `eps50`
    - `soilType = 2` for sand layers using `phi_deg` and `gamma`
    - The last layer bottom depth must match the pile length
    - `pult` and `y50` are derived internally for PySimple1
    """
)

if run_btn:
    try:
        config = {
            "head_condition": head_condition,
            "base_condition": base_condition,
            "head_lateral_load": float(head_lateral_load),
            "head_axial_load": float(head_axial_load),
            "head_moment": float(head_moment),
            "pile_length": float(pile_length),
            "n_ele": int(n_ele),
            "pile_diameter": float(pile_diameter),
            "e_pile": float(e_pile),
            "pysimple1_c": float(pysimple1_c),
            "cd_default": float(cd_default),
            "spring_sign": float(spring_sign),
        }

        clean_soil_layers = validate_soil_layers(soil_layers, float(pile_length))
        results = run_analysis(config, clean_soil_layers)
        summary = results["summary"]

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Head disp (mm)", f"{summary['head_disp_mm']:.3f}")
        col2.metric("Max shear (kN)", f"{summary['max_shear_kN']:.3f}")
        col3.metric("Max moment (kN.m)", f"{summary['max_moment_kNm']:.3f}")
        col4.metric("Bottom moment (kN.m)", f"{summary['bottom_moment_kNm']:.6f}")

        st.subheader("Model summary")
        st.write({
            "Head condition": head_condition,
            "Base condition": base_condition,
            "Pile length (m)": pile_length,
            "Pile diameter (m)": pile_diameter,
            "Elements": int(n_ele),
            "Head lateral load (kN)": head_lateral_load / 1e3,
            "Head axial load (kN)": head_axial_load / 1e3,
            "Head moment (kN.m)": head_moment / 1e3,
            "Head rotation (rad)": summary["head_rot_rad"],
            "Base rotation (rad)": summary["base_rot_rad"],
            "Max soil reaction (kN)": summary["max_soil_reaction_kN"],
            "Top moment (kN.m)": summary["top_moment_kNm"],
            "Bottom moment (kN.m)": summary["bottom_moment_kNm"],
        })

        st.subheader("Pile and soil profile after full load")
        st.pyplot(plot_pile_soil_profile(
            results["depth"],
            results["ux"],
            clean_soil_layers,
            float(pile_diameter),
            f"Pile deformation within soil layers ({head_condition}-head, {base_condition}-base)"
        ))

        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(plot_deflection(
                results["depth"],
                results["ux"],
                f"Pile lateral deflection ({head_condition}-head, {base_condition}-base)"
            ))
            st.pyplot(plot_shear_profile(
                results["force_data"],
                f"Pile shear profile ({head_condition}-head, {base_condition}-base)"
            ))

        with c2:
            st.pyplot(plot_soil_reaction_profile(
                results["z_spring"],
                results["p_spring"],
                f"Soil reaction profile ({head_condition}-head, {base_condition}-base)"
            ))
            st.pyplot(plot_moment_profile(
                results["force_data"],
                f"Pile bending moment profile ({head_condition}-head, {base_condition}-base)"
            ))

        st.subheader("Spring table")
        st.dataframe(results["spring_df"], use_container_width=True)

        st.subheader("Node force table")
        st.dataframe(results["node_force_df"], use_container_width=True)

        st.subheader("Element end force table")
        st.dataframe(results["element_df"], use_container_width=True)

        csv_node = results["node_force_df"].to_csv(index=False).encode("utf-8")
        csv_ele = results["element_df"].to_csv(index=False).encode("utf-8")
        csv_spring = results["spring_df"].to_csv(index=False).encode("utf-8")

        d1, d2, d3 = st.columns(3)
        d1.download_button("Download node force CSV", csv_node, file_name="node_force_table.csv", mime="text/csv")
        d2.download_button("Download element force CSV", csv_ele, file_name="element_force_table.csv", mime="text/csv")
        d3.download_button("Download spring CSV", csv_spring, file_name="spring_table.csv", mime="text/csv")

    except Exception as e:
        st.exception(e)
else:
    st.info("Set the inputs, edit the soil layers in the GUI, and click Run analysis.")
