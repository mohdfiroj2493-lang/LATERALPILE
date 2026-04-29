import math
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import openseespy.opensees as op
    OPENSEES_AVAILABLE = True
except Exception:
    OPENSEES_AVAILABLE = False

# ============================================================
# BNWF SINGLE PILE STREAMLIT APP
# Built directly from the user-provided OpenSeesPy script
# Basic units are kN and m
# ============================================================

st.set_page_config(page_title="BNWF Single Pile", layout="wide")
st.title("BNWF Single Pile Analysis")


# ============================================================
# PARAMETER FUNCTIONS - DIRECTLY FROM USER CODE
# ============================================================
def get_pyParam(pyDepth, gamma, phiDegree, b, pEleLength, puSwitch, kSwitch, gwtSwitch):
    pi = 3.14159265358979
    phi = phiDegree * (pi / 180)
    zbRatio = pyDepth / b

    if puSwitch == 1:
        zb = []
        dataNum = 41
        for i in range(dataNum):
            b1 = i * 0.125
            zb.append(b1)
        As = [
            2.8460, 2.7105, 2.6242, 2.5257, 2.4271, 2.3409, 2.2546, 2.1437, 2.0575, 1.9589,
            1.8973, 1.8111, 1.7372, 1.6632, 1.5893, 1.5277, 1.4415, 1.3799, 1.3368, 1.2690,
            1.2074, 1.1581, 1.1211, 1.0780, 1.0349, 1.0164, 0.9979, 0.9733, 0.9610, 0.9487,
            0.9363, 0.9117, 0.8994, 0.8994, 0.8871, 0.8871, 0.8809, 0.8809, 0.8809, 0.8809, 0.8809
        ]

        A = 0.88
        for i in range(dataNum - 1):
            if zbRatio >= 5.0:
                A = 0.88
            elif zb[i] <= zbRatio <= zb[i + 1]:
                A = (As[i + 1] - As[i]) / (zb[i + 1] - zb[i]) * (zbRatio - zb[i]) + As[i]
                break

        alpha = phi / 2
        beta = pi / 4 + phi / 2
        K0 = 0.4

        tan_1 = math.tan(pi / 4 - phi / 2)
        Ka = math.pow(tan_1, 2)

        tan_2 = math.tan(phi)
        tan_3 = math.tan(beta - phi)
        sin_1 = math.sin(beta)
        cos_1 = math.cos(alpha)
        c1 = K0 * tan_2 * sin_1 / (tan_3 * cos_1)

        tan_4 = math.tan(beta)
        tan_5 = math.tan(alpha)
        c2 = (tan_4 / tan_3) * tan_4 * tan_5
        c3 = K0 * tan_4 * (tan_2 * sin_1 - tan_5)
        c4 = tan_4 / tan_3 - Ka

        pow_1 = math.pow(tan_4, 8)
        pow_2 = math.pow(tan_4, 4)
        c5 = Ka * (pow_1 - 1)
        c6 = K0 * tan_2 * pow_2

        pst = gamma * pyDepth * (pyDepth * (c1 + c2 + c3) + b * c4)
        psd = b * gamma * pyDepth * (c5 + c6)

        if pst <= psd:
            if pyDepth == 0:
                pu = 0.01
            else:
                pu = A * pst
        else:
            pu = A * psd

        pult = pu * pEleLength

    elif puSwitch == 2:
        cos_2 = math.cos(phi)
        tan_6 = math.tan(pi / 4 + phi / 2)
        sin_2 = math.sin(phi)
        sin_3 = math.sin(pi / 4 + phi / 2)

        tan_2 = math.tan(phi)
        tan_1 = math.tan(pi / 4 - phi / 2)
        exp_1 = math.exp((pi / 2 + phi) * tan_2)
        exp_2 = math.exp(-(pi / 2 - phi) * tan_2)

        Kqo = exp_1 * cos_2 * tan_6 - exp_2 * cos_2 * tan_1
        Kco = (1 / tan_2) * (exp_1 * cos_2 * tan_6 - 1)

        exp_3 = math.exp(pi * tan_2)
        pow_3 = math.pow(tan_2, 4)
        pow_4 = math.pow(tan_6, 2)
        dcinf = 1.58 + 4.09 * pow_3
        Nc = (1 / tan_2) * exp_3 * (pow_4 - 1)
        Ko = 1 - sin_2
        Kcinf = Nc * dcinf
        Kqinf = Kcinf * Ko * tan_2

        aq = (Kqo / (Kqinf - Kqo)) * (Ko * sin_2 / sin_3)
        KqD = (Kqo + Kqinf * aq * zbRatio) / (1 + aq * zbRatio)

        if pyDepth == 0:
            pu = 0.01
        else:
            pu = gamma * pyDepth * KqD * b

        pult = pu * pEleLength
        A = 1.0
    else:
        raise ValueError("puSwitch must be 1 or 2")

    ph = [28.8, 29.5, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0]
    if gwtSwitch == 1:
        k = [10, 23, 45, 61, 80, 100, 120, 140, 160, 182, 215, 250, 275]
    else:
        k = [10, 20, 33, 42, 50, 60, 70, 85, 95, 107, 122, 141, 155]

    khat = k[0]
    dataNum = 13
    for i in range(dataNum - 1):
        if ph[i] <= phiDegree <= ph[i + 1]:
            khat = (k[i + 1] - k[i]) / (ph[i + 1] - ph[i]) * (phiDegree - ph[i]) + k[i]
            break
        elif phiDegree < ph[0]:
            khat = k[0]
        elif phiDegree > ph[-1]:
            khat = k[-1]

    k_SIunits = khat * 271.45
    sigV = pyDepth * gamma
    if sigV == 0:
        sigV = 0.01
    if kSwitch == 2:
        cSigma = math.pow(50 / sigV, 0.5)
        k_SIunits = cSigma * k_SIunits

    x = 0.5
    log_1 = math.log((1 + x) / (1 - x))
    atanh_value = 0.5 * log_1

    if pyDepth == 0.0:
        pyDepth = 0.01

    y50 = 0.5 * (pu / A) / (k_SIunits * pyDepth) * atanh_value
    return [pult, y50]


def get_qzParam(phiDegree, b, sigV, G):
    pi = 3.14159265358979
    sin_4 = math.sin(phiDegree * (pi / 180))
    Ko = 1 - sin_4

    phi = phiDegree * (pi / 180)
    tan_7 = math.tan(phi)
    Ir = G / (sigV * tan_7)
    tan_8 = math.tan(pi / 4 + phi / 2)
    sin_5 = math.sin(phi)
    pow_4 = math.pow(tan_8, 2)
    pow_5 = math.pow(Ir, (4 * sin_5) / (3 * (1 + sin_5)))
    exp_4 = math.exp(pi / 2 - phi)

    Nq = (1 + 2 * Ko) * (1 / (3 - sin_5)) * exp_4 * pow_4 * pow_5
    qu = Nq * sigV
    pow_6 = math.pow(b, 2)
    qult = qu * pi * pow_6 / 4

    zc = 0.05 * b
    z50 = 0.125 * zc
    return [qult, z50]


def get_tzParam(phi, b, sigV, pEleLength):
    pi = 3.14159265358979
    delta = 0.8 * phi * pi / 180

    if sigV == 0.0:
        sigV = 0.01

    tan_9 = math.tan(delta)
    tu = 0.4 * sigV * pi * b * tan_9
    tult = tu * pEleLength

    kf = [6000, 10000, 10000, 14000, 14000, 18000]
    fric = [28, 31, 32, 34, 35, 38]
    dataNum = len(fric)

    if phi < fric[0]:
        k = kf[0]
    elif phi > fric[5]:
        k = kf[5]
    else:
        k = kf[0]
        for i in range(dataNum - 1):
            if fric[i] <= phi <= fric[i + 1]:
                k = ((kf[i + 1] - kf[i]) / (fric[i + 1] - fric[i])) * (phi - fric[i]) + kf[i]
                break

    kSIunits = k * 1.885
    z50 = tult / kSIunits
    return [tult, z50]


# ============================================================
# MODEL BUILD AND ANALYSIS - DIRECTLY FROM USER CODE
# ============================================================
def get_layer_at_depth(depth, soil_layers):
    for layer in soil_layers:
        if layer["z_top"] <= depth <= layer["z_bot"]:
            return layer
    return soil_layers[-1]


def build_and_run_model(params: Dict[str, float], soil_layers):
    op.wipe()

    L1 = params["L1"]
    L2 = params["L2"]
    diameter = params["diameter"]
    nElePile = int(params["nElePile"])
    eleSize = (L1 + L2) / nElePile
    nNodePile = 1 + nElePile

    default_gamma = params.get("gamma", 17.0)
    default_phi = params.get("phi", 36.0)
    default_Gsoil = params.get("Gsoil", 150000.0)
    puSwitch = int(params["puSwitch"])
    kSwitch = int(params["kSwitch"])
    gwtSwitch = int(params["gwtSwitch"])

    # ----------------------------------------------------------
    # create spring nodes
    # ----------------------------------------------------------
    op.model('basic', '-ndm', 3, '-ndf', 3)
    count = 0
    for i in range(nNodePile):
        zCoord = eleSize * i
        if zCoord <= L2:
            op.node(i + 1, 0.0, 0.0, zCoord)
            op.node(i + 101, 0.0, 0.0, zCoord)
            count += 1
    nNodeEmbed = count

    for i in range(nNodeEmbed):
        op.fix(i + 1, 1, 1, 1)
        op.fix(i + 101, 0, 1, 1)

    # ----------------------------------------------------------
    # spring materials
    # ----------------------------------------------------------
    for i in range(1, nNodeEmbed + 1):
        pyDepth = L2 - eleSize * (i - 1)
        layer_depth = L2 - pyDepth
        layer = get_layer_at_depth(layer_depth, soil_layers)
        gamma = layer.get("gamma", default_gamma)
        phi = layer.get("phi", default_phi)
        soilType = layer.get("soilType",2)
        # For now, both clay and sand use same py function; extend later if needed
        pyParam = get_pyParam(pyDepth, gamma, phi, diameter, eleSize, puSwitch, kSwitch, gwtSwitch)
        pult = pyParam[0]
        y50 = pyParam[1]
        op.uniaxialMaterial('PySimple1', i, 2, pult, y50, 0.0)

    for i in range(2, nNodeEmbed + 1):
        pyDepth = eleSize * (i - 1)
        layer = get_layer_at_depth(pyDepth, soil_layers)
        gamma = layer.get("gamma", default_gamma)
        phi = layer.get("phi", default_phi)
        soilType = layer.get("soilType",2)
        sigV = gamma * pyDepth
        tzParam = get_tzParam(phi, diameter, sigV, eleSize)
        tult = tzParam[0]
        z50 = tzParam[1]
        op.uniaxialMaterial('TzSimple1', i + 100, 2, tult, z50, 0.0)

    tip_layer = get_layer_at_depth(L2, soil_layers)
    gamma = tip_layer.get("gamma", default_gamma)
    phi = tip_layer.get("phi", default_phi)
    Gsoil = tip_layer.get("Gsoil", default_Gsoil)
    sigVq = gamma * L2
    qzParam = get_qzParam(phi, diameter, sigVq, Gsoil)
    qult = qzParam[0]
    z50q = qzParam[1]
    op.uniaxialMaterial('TzSimple1', 101, 2, qult, z50q, 0.0)

    # ----------------------------------------------------------
    # zero-length spring elements
    # ----------------------------------------------------------
    op.element('zeroLength', 1001, 1, 101, '-mat', 1, 101, '-dir', 1, 3)
    for i in range(2, nNodeEmbed + 1):
        op.element('zeroLength', 1000 + i, i, 100 + i, '-mat', i, 100 + i, '-dir', 1, 3)

    # ----------------------------------------------------------
    # create pile nodes
    # ----------------------------------------------------------
    op.model('basic', '-ndm', 3, '-ndf', 6)
    for i in range(1, nNodePile + 1):
        # node 201 starts at the ground line, increasing downward
        zCoord = eleSize * (i - 1)
        op.node(i + 200, 0.0, 0.0, zCoord)

    op.geomTransf('Linear', 1, 0.0, -1.0, 0.0)

    # pile head boundary condition
    # For lateral pile analysis, "free" vs "fixed" refers to head rotation,
    # not whether the loaded head node can translate laterally.
    if params["head_condition"] == "fixed":
        # translation in global X remains free; rotation about global Y is fixed
        op.fix(200 + nNodePile, 0, 1, 0, 1, 1, 1)
    else:
        # translation in global X and rotation about global Y are both free
        op.fix(200 + nNodePile, 0, 1, 0, 1, 0, 1)

    for i in range(201, 200 + nNodePile):
        op.fix(i, 0, 1, 0, 1, 0, 1)

    for i in range(1, nNodeEmbed + 1):
        op.equalDOF(200 + i, 100 + i, 1, 3)

    # ----------------------------------------------------------
    # pile section
    # ----------------------------------------------------------
    secTag3D = 3
    E = params["E"]
    A = params["A"]
    Iz = params["Iz"]
    Iy = params["Iy"]
    G = params["G"]
    J = params["J"]

    op.section('Elastic', 1, E, A, Iz, Iy, G, J)
    op.uniaxialMaterial('Elastic', 3000, 1e10)
    op.section('Aggregator', secTag3D, 3000, 'T', '-section', 1)

    op.beamIntegration('Legendre', 1, secTag3D, 3)
    for i in range(201, 201 + nElePile):
        op.element('dispBeamColumn', i, i, i + 1, 1, 1)

    # ----------------------------------------------------------
    # loading
    # ----------------------------------------------------------
    op.setTime(10.0)
    values = [0.0, 0.0, 1.0, 1.0]
    time = [0.0, 10.0, 20.0, 10000.0]
    nodeTag = 200 + nNodePile
    # load vector for 3D beam-column node: Fx, Fy, Fz, Mx, My, Mz
    loadValues = [params["head_load_x"], 0.0, 0.0, 0.0, params["head_moment"], 0.0]
    op.timeSeries('Path', 1, '-values', *values, '-time', *time, '-factor', 1.0)
    op.pattern('Plain', 10, 1)
    op.load(nodeTag, *loadValues)

    # ----------------------------------------------------------
    # analysis
    # ----------------------------------------------------------
    op.integrator('LoadControl', params["load_increment"])
    op.numberer('RCM')
    op.system('SparseGeneral')
    op.constraints('Transformation')
    op.test('NormDispIncr', 1e-5, 20, 1)
    op.algorithm('Newton')
    op.analysis('Static')

    ok = op.analyze(int(params["n_steps"]))
    op.reactions()

    # ----------------------------------------------------------
    # results
    # ----------------------------------------------------------
    disp_rows = []
    for i in range(201, nodeTag + 1):
        coord = op.nodeCoord(i)
        disp = op.nodeDisp(i)
        react = op.nodeReaction(i)
        # Convert OpenSees z-coordinate to depth measured downward from pile head
        depth_from_head = (L1 + L2) - coord[2]
        disp_rows.append({
            "node": i,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2],
            "depth": depth_from_head,
            "ux": disp[0],
            "uy": disp[1],
            "uz": disp[2],
            "rx": disp[3],
            "ry": disp[4],
            "rz": disp[5],
            "Rx": react[0] if len(react) > 0 else 0.0,
            "Ry": react[1] if len(react) > 1 else 0.0,
            "Rz": react[2] if len(react) > 2 else 0.0,
        })
    node_df = pd.DataFrame(disp_rows)

    spring_rows = []
    for i in range(1, nNodeEmbed + 1):
        coord = op.nodeCoord(i)
        react = op.nodeReaction(i)
        depth_from_head = (L1 + L2) - coord[2]
        spring_rows.append({
            "spring_node": i,
            "z": coord[2],
            "depth": depth_from_head,
            "Rx": react[0] if len(react) > 0 else 0.0,
            "Ry": react[1] if len(react) > 1 else 0.0,
            "Rz": react[2] if len(react) > 2 else 0.0,
        })
    spring_df = pd.DataFrame(spring_rows)

    ele_rows = []
    for i in range(201, 201 + nElePile):
        force = op.eleForce(i)
        zi = op.nodeCoord(i)[2]
        zj = op.nodeCoord(i + 1)[2]
        ele_rows.append({
            "element": i,
            "zi": zi,
            "zj": zj,
            "P_i": force[0],
            "Vy_i": force[1],
            "Vz_i": force[2],
            "T_i": force[3],
            "My_i": force[4],
            "Mz_i": force[5],
            "P_j": force[6],
            "Vy_j": force[7],
            "Vz_j": force[8],
            "T_j": force[9],
            "My_j": force[10],
            "Mz_j": force[11],
        })
    ele_df = pd.DataFrame(ele_rows)

    return ok, node_df, spring_df, ele_df


# ============================================================
# PLOTTING
# ============================================================
def plot_displacement_profile(x, depth):
    depth = np.asarray(depth, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(depth)
    depth = depth[order]
    x = x[order]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(x, depth, lw=2.2)
    ax.set_ylim(max(depth) + 1.0, -1.0)
    ax.set_ylabel("Depth from pile head (m)")
    ax.set_xlabel("ux (m)")
    ax.set_title("Lateral displacement", fontsize=18)
    ax.grid(True, alpha=0.35)
    return fig


def plot_reaction_profile(x, depth):
    depth = np.asarray(depth, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(depth)
    depth = depth[order]
    x = x[order]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(x, depth, lw=2.2)
    ax.set_ylim(max(depth) + 1.0, -1.0)
    ax.set_ylabel("Depth from pile head (m)")
    ax.set_xlabel("Reaction Rx (kN)")
    ax.set_title("Spring reaction profile", fontsize=18)
    ax.grid(True, alpha=0.35)
    return fig


def plot_moment_profile(x, depth):
    depth = np.asarray(depth, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(depth)
    depth = depth[order]
    x = x[order]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(x, depth, lw=2.2)
    ax.set_ylim(max(depth) + 1.0, -1.0)
    ax.set_ylabel("Depth from pile head (m)")
    ax.set_xlabel("My (kN.m)")
    ax.set_title("Bending moment profile", fontsize=18)
    ax.grid(True, alpha=0.35)
    return fig


def plot_shear_profile(x, depth):
    depth = np.asarray(depth, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(depth)
    depth = depth[order]
    x = x[order]

    fig, ax = plt.subplots(figsize=(7, 9))
    ax.plot(x, depth, lw=2.2)
    ax.set_ylim(max(depth) + 1.0, -1.0)
    ax.set_ylabel("Depth from pile head (m)")
    ax.set_xlabel("Fx (kN)")
    ax.set_title("Shear profile", fontsize=18)
    ax.grid(True, alpha=0.35)
    return fig


def plot_deformed_shape(node_df, L1, L2):
    fig, ax = plt.subplots(figsize=(7, 9))
    df = node_df.sort_values("depth").copy()
    depth = df["depth"].values
    ux = df["ux"].values
    max_abs = max(np.max(np.abs(ux)), 1e-12)
    scale = min(50.0, max(1.0, 0.15 * max(depth) / max_abs))

    # Embedded zone measured from pile head starts at L1
    ax.axhspan(L1, L1 + L2, color="#d6eaf8", alpha=0.35)
    ax.plot(np.zeros_like(depth), depth, "k--", lw=1.2, label="Undeformed")
    ax.plot(ux * scale, depth, "b-", lw=2.4, label=f"Deformed x{scale:.1f}")
    ax.set_ylim(max(depth) + 1.0, -1.0)
    ax.set_xlabel("Horizontal displacement (scaled)")
    ax.set_ylabel("Depth from pile head (m)")
    ax.set_title("Pile deformed shape", fontsize=18)
    ax.grid(True, alpha=0.35)
    ax.legend()
    return fig


# ============================================================
# UI
# ============================================================
st.sidebar.header("Pile Geometry")
L1 = st.sidebar.number_input("Length above ground L1 (m)", value=1.0, step=0.1)
L2 = st.sidebar.number_input("Embedded length L2 (m)", value=20.0, step=0.5)
diameter = st.sidebar.number_input("Pile diameter (m)", value=1.0, step=0.05)
nElePile = st.sidebar.number_input("Number of pile elements", value=84, min_value=4, step=2)

st.sidebar.header("Soil Properties")
puSwitch = st.sidebar.selectbox("pult method", [1, 2], index=0, format_func=lambda x: "API" if x == 1 else "Brinch Hansen")
kSwitch = st.sidebar.selectbox("k variation", [1, 2], index=0, format_func=lambda x: "API linear" if x == 1 else "Modified API parabolic")
gwtSwitch = st.sidebar.selectbox("Groundwater switch", [1, 2], index=0, format_func=lambda x: "Above GWT" if x == 1 else "Below GWT")

st.subheader("Soil Layers")
if "soil_layers" not in st.session_state:
    st.session_state.soil_layers = [
        {"name": "Layer 1", "z_top": 0.0, "z_bot": 20.0, "gamma": 17.0, "phi": 36.0, "Gsoil": 150000.0}
    ]

col_add1, col_add2 = st.columns([1, 5])
with col_add1:
    if st.button("Add layer"):
        next_top = st.session_state.soil_layers[-1]["z_bot"] if st.session_state.soil_layers else 0.0
        st.session_state.soil_layers.append({
            "name": f"Layer {len(st.session_state.soil_layers)+1}",
            "z_top": float(next_top),
            "z_bot": float(next_top) + 1.0,
            "gamma": 17.0,
            "phi": 36.0,
            "Gsoil": 150000.0,
        })

updated_layers = []
# Soil type selection added: 1 = clay, 2 = sand
for i, layer in enumerate(st.session_state.soil_layers):
    with st.expander(f"Soil Layer {i+1}", expanded=True):
        c1, c2, c3 = st.columns(3)
        name = c1.text_input("Name", value=layer.get("name", f"Layer {i+1}"), key=f"lname_{i}")
        z_top = c2.number_input("z_top (m)", value=float(layer.get("z_top", 0.0)), step=0.5, key=f"ztop_{i}")
        z_bot = c3.number_input("z_bot (m)", value=float(layer.get("z_bot", 1.0)), step=0.5, key=f"zbot_{i}")
        c4, c5, c6, c7 = st.columns(4)
        soil_type = c4.selectbox(
            "Soil Type",
            ["Sand", "Clay"],
            index=0 if layer.get("soilType", 2) == 2 else 1,
            key=f"stype_{i}",
        )
        gamma_i = c5.number_input(
            "gamma (kN/m3)",
            value=float(layer.get("gamma", 17.0)),
            step=0.5,
            key=f"gamma_{i}",
        )

        # Sand uses phi, Clay uses cohesion c
        if soil_type == "Sand":
            phi_i = c6.number_input(
                "phi (deg)",
                value=float(layer.get("phi", 36.0)),
                step=1.0,
                key=f"phi_{i}",
            )
            c_i = layer.get("c", 0.0)
        else:
            c_i = c6.number_input(
                "c (kPa)",
                value=float(layer.get("c", 25.0)),
                step=5.0,
                key=f"c_{i}",
            )
            phi_i = layer.get("phi", 0.0)

        gsoil_i = c7.number_input(
            "Gsoil (kPa)",
            value=float(layer.get("Gsoil", 150000.0)),
            step=1000.0,
            key=f"gsoil_{i}",
        )(
            "Gsoil (kPa)",
            value=float(layer.get("Gsoil", 150000.0)),
            step=1000.0,
            key=f"gsoil_{i}",
        )
        if st.button(f"Delete layer {i+1}", key=f"del_layer_{i}"):
            continue
        updated_layers.append({
            "name": name,
            "z_top": float(z_top),
            "z_bot": float(z_bot),
            "gamma": float(gamma_i),
            "phi": float(phi_i),
            "c": float(c_i),
            "Gsoil": float(gsoil_i),
            "soilType": 2 if soil_type == "Sand" else 1
        })

st.session_state.soil_layers = sorted(updated_layers, key=lambda x: x["z_top"])
soil_layers = st.session_state.soil_layers
st.dataframe(pd.DataFrame(soil_layers), use_container_width=True)

st.sidebar.header("Pile Section")
E = st.sidebar.number_input("E", value=25000000.0, step=1e6, format="%.6e")
A = st.sidebar.number_input("A", value=0.785, step=0.01)
Iz = st.sidebar.number_input("Iz", value=0.049, step=0.001)
Iy = st.sidebar.number_input("Iy", value=0.049, step=0.001)
G = st.sidebar.number_input("G", value=9615385.0, step=1e6, format="%.6e")
J = st.sidebar.number_input("J", value=0.098, step=0.001)

st.sidebar.header("Head Boundary Condition")
head_condition = st.sidebar.selectbox("Top of pile", ["free", "fixed"], index=0)
head_moment = st.sidebar.number_input("Head moment My (kN.m)", value=0.0, step=10.0)

st.sidebar.header("Loading and Analysis")
head_load_x = st.sidebar.number_input("Head load in x-direction (kN)", value=3500.0, step=100.0)
load_increment = st.sidebar.number_input("LoadControl increment", value=0.05, step=0.01, format="%.3f")
n_steps = st.sidebar.number_input("Number of analysis steps", value=201, min_value=1, step=10)

params = {
    "L1": float(L1),
    "L2": float(L2),
    "diameter": float(diameter),
    "nElePile": int(nElePile),
    "puSwitch": int(puSwitch),
    "kSwitch": int(kSwitch),
    "gwtSwitch": int(gwtSwitch),
    "E": float(E),
    "A": float(A),
    "Iz": float(Iz),
    "Iy": float(Iy),
    "G": float(G),
    "J": float(J),
    "head_load_x": float(head_load_x),
    "head_moment": float(head_moment),
    "head_condition": str(head_condition),
    "load_increment": float(load_increment),
    "n_steps": int(n_steps),
}

st.subheader("Model summary")
st.dataframe(pd.DataFrame([
    {"Parameter": k, "Value": v} for k, v in params.items()
]), use_container_width=True)

run_clicked = st.button("Run analysis", type="primary")

if not OPENSEES_AVAILABLE:
    st.warning("OpenSeesPy is not installed in this environment.")

if run_clicked and OPENSEES_AVAILABLE:
    try:
        ok, node_df, spring_df, ele_df = build_and_run_model(params, soil_layers)

        if ok != 0:
            st.error("Analysis did not converge.")
        else:
            top = node_df.iloc[-1]
            depth_ele = (params["L1"] + params["L2"]) - 0.5 * (ele_df["zi"].values + ele_df["zj"].values)

            # For a vertical pile with lateral load in global X:
            # shear is the global X force component and bending is about global Y.
            # Use a consistent single element end for diagrams.
            # Averaging i-end and j-end forces cancels shear and distorts moment because
            # the two ends are equal-and-opposite nodal actions for the same element.
            shear_x = ele_df["P_i"].to_numpy(dtype=float)
            moment_y = ele_df["My_i"].to_numpy(dtype=float)

            node_plot = node_df.copy().sort_values("depth")
            spring_plot = spring_df.copy().sort_values("depth")

            c1, c2, c3 = st.columns(3)
            c1.metric("Top displacement ux", f"{top['ux']:.6f} m")
            c2.metric("Max |My|", f"{np.max(np.abs(moment_y)):.3f} kN.m")
            c3.metric("Max |Fx|", f"{np.max(np.abs(shear_x)):.3f} kN")

            st.pyplot(plot_deformed_shape(node_plot, params["L1"], params["L2"]))

            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(plot_displacement_profile(node_plot["ux"].values, node_plot["depth"].values))
                st.pyplot(plot_reaction_profile(spring_plot["Rx"].values, spring_plot["depth"].values))
            with p2:
                st.pyplot(plot_moment_profile(moment_y, depth_ele))
                st.pyplot(plot_shear_profile(shear_x, depth_ele))

            st.subheader("Pile node results")
            st.dataframe(node_df, use_container_width=True)
            st.subheader("Spring reactions")
            st.dataframe(spring_df, use_container_width=True)
            st.subheader("Pile element forces")
            st.dataframe(ele_df, use_container_width=True)

    except Exception as exc:
        st.exception(exc)
