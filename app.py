import math
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
except Exception:
    OPENSEES_AVAILABLE = False

# ============================================================
# BNWF SINGLE PILE APP
# Rebuilt from uploaded Tcl files:
# - staticBNWFsingle.tcl
# - get_pyParam.tcl
# - get_tzParam.tcl
# - get_qzParam.tcl
# - elasticPileSection.tcl
# ============================================================

st.set_page_config(page_title="BNWF Single Pile", layout="wide")
st.title("BNWF Single Pile Analysis")


# ============================================================
# DEFAULTS FROM THE TCL MODEL
# ============================================================
DEFAULTS = {
    "L1": 1.0,
    "L2": 20.0,
    "diameter": 1.0,
    "nElePile": 84,
    "gamma": 17.0,
    "phi": 36.0,
    "Gsoil": 150000.0,
    "puSwitch": 1,
    "kSwitch": 1,
    "gwtSwitch": 1,
    "head_load_x": 3500.0,
    "load_steps": 201,
    "load_increment": 0.05,
    "pult_cd": 0.0,
    "tz_cd": 0.0,
    "qz_suction": 0.0,
    "qz_cd": 0.0,
    # elasticPileSection.tcl
    "E": 25000000.0,
    "A": 0.785,
    "Iz": 0.049,
    "Iy": 0.049,
    "G": 9615385.0,
    "J": 0.098,
    "torsion_stiffness": 1.0e10,
}


# ============================================================
# PARAMETER FUNCTIONS TRANSLATED FROM TCL
# ============================================================
def interp_piecewise(x: float, xs: List[float], ys: List[float], x_left=None, x_right=None) -> float:
    if x <= xs[0]:
        return ys[0] if x_left is None else x_left
    if x >= xs[-1]:
        return ys[-1] if x_right is None else x_right
    return float(np.interp(x, xs, ys))


def get_py_param(py_depth: float, gamma: float, phi_degree: float, b: float, p_ele_length: float,
                 pu_switch: int, k_switch: int, gwt_switch: int) -> Tuple[float, float]:
    pi = math.pi
    phi = math.radians(phi_degree)
    zb_ratio = py_depth / b if b > 0 else 0.0

    if pu_switch == 1:
        zb_vals = [
            0.0000,0.1250,0.2500,0.3750,0.5000,0.6250,0.7500,0.8750,1.0000,1.1250,
            1.2500,1.3750,1.5000,1.6250,1.7500,1.8750,2.0000,2.1250,2.2500,2.3750,
            2.5000,2.6250,2.7500,2.8750,3.0000,3.1250,3.2500,3.3750,3.5000,3.6250,
            3.7500,3.8750,4.0000,4.1250,4.2500,4.3750,4.5000,4.6250,4.7500,4.8750,5.0000
        ]
        A_vals = [
            2.8460,2.7105,2.6242,2.5257,2.4271,2.3409,2.2546,2.1437,2.0575,1.9589,
            1.8973,1.8111,1.7372,1.6632,1.5893,1.5277,1.4415,1.3799,1.3368,1.2690,
            1.2074,1.1581,1.1211,1.0780,1.0349,1.0164,0.9979,0.9733,0.9610,0.9487,
            0.9363,0.9117,0.8994,0.8994,0.8871,0.8871,0.8809,0.8809,0.8809,0.8809,0.8809
        ]
        A = 0.88 if zb_ratio >= 5.0 else float(np.interp(zb_ratio, zb_vals, A_vals))

        alpha = phi / 2.0
        beta = pi / 4.0 + phi / 2.0
        K0 = 0.4
        Ka = math.tan(pi / 4.0 - phi / 2.0) ** 2

        c1 = K0 * math.tan(phi) * math.sin(beta) / (math.tan(beta - phi) * math.cos(alpha))
        c2 = math.tan(beta) / math.tan(beta - phi) * math.tan(beta) * math.tan(alpha)
        c3 = K0 * math.tan(beta) * (math.tan(phi) * math.sin(beta) - math.tan(alpha))
        c4 = math.tan(beta) / math.tan(beta - phi) - Ka
        c5 = Ka * (math.tan(beta) ** 8 - 1.0)
        c6 = K0 * math.tan(phi) * (math.tan(beta) ** 4)

        pst = gamma * py_depth * (py_depth * (c1 + c2 + c3) + b * c4)
        psd = b * gamma * py_depth * (c5 + c6)

        if pst <= psd:
            pu = 0.01 if py_depth == 0 else A * pst
        else:
            pu = A * psd

        pult = pu * p_ele_length

    else:
        Kqo = math.exp((pi / 2 + phi) * math.tan(phi)) * math.cos(phi) * math.tan(pi / 4 + phi / 2) \
            - math.exp(-(pi / 2 - phi) * math.tan(phi)) * math.cos(phi) * math.tan(pi / 4 - phi / 2)
        Kco = (1 / math.tan(phi)) * (
            math.exp((pi / 2 + phi) * math.tan(phi)) * math.cos(phi) * math.tan(pi / 4 + phi / 2) - 1
        )
        dcinf = 1.58 + 4.09 * (math.tan(phi) ** 4)
        Nc = (1 / math.tan(phi)) * math.exp(pi * math.tan(phi)) * ((math.tan(pi / 4 + phi / 2) ** 2) - 1)
        Ko = 1 - math.sin(phi)
        Kcinf = Nc * dcinf
        Kqinf = Kcinf * Ko * math.tan(phi)
        aq = (Kqo / (Kqinf - Kqo)) * (Ko * math.sin(phi) / math.sin(pi / 4 + phi / 2))
        KqD = (Kqo + Kqinf * aq * zb_ratio) / (1 + aq * zb_ratio)
        pu = 0.01 if py_depth == 0 else gamma * py_depth * KqD * b
        pult = pu * p_ele_length
        A = 1.0

    ph = [28.8,29.5,30.0,31.0,32.0,33.0,34.0,35.0,36.0,37.0,38.0,39.0,40.0]
    if gwt_switch == 1:
        ktab = [10,23,45,61,80,100,120,140,160,182,215,250,275]
    else:
        ktab = [10,20,33,42,50,60,70,85,95,107,122,141,155]

    khat = interp_piecewise(phi_degree, ph, ktab, x_left=ktab[0], x_right=ktab[-1])
    k_si = khat * 271.45

    sigV = py_depth * gamma
    if sigV == 0:
        sigV = 0.01
    if k_switch == 2:
        c_sigma = (50 / sigV) ** 0.5
        k_si = c_sigma * k_si

    x = 0.5
    atanh_value = 0.5 * math.log((1 + x) / (1 - x))
    py_depth_eff = 0.01 if py_depth == 0.0 else py_depth
    y50 = 0.5 * (pu / A) / (k_si * py_depth_eff) * atanh_value
    return pult, y50


def get_tz_param(phi: float, b: float, sigV: float, p_ele_length: float) -> Tuple[float, float]:
    pi = math.pi
    delta = 0.8 * phi * pi / 180.0
    if sigV == 0.0:
        sigV = 0.01
    tu = 0.4 * sigV * pi * b * math.tan(delta)
    tult = tu * p_ele_length

    kf = [6000, 10000, 10000, 14000, 14000, 18000]
    fric = [28, 31, 32, 34, 35, 38]
    k = interp_piecewise(phi, fric, kf, x_left=kf[0], x_right=kf[-1])
    k_si = k * 1.885
    z50 = tult / k_si
    return tult, z50


def get_qz_param(phi_degree: float, b: float, sigV: float, G: float) -> Tuple[float, float]:
    pi = math.pi
    Ko = 1 - math.sin(math.radians(phi_degree))
    phi = math.radians(phi_degree)
    Ir = G / (sigV * math.tan(phi))
    Nq = (1 + 2 * Ko) * (1 / (3 - math.sin(phi))) * math.exp(pi / 2 - phi) * (math.tan(pi / 4 + phi / 2) ** 2) \
        * (Ir ** ((4 * math.sin(phi)) / (3 * (1 + math.sin(phi)))))
    qu = Nq * sigV
    qult = qu * pi * b**2 / 4
    zc = 0.05 * b
    z50 = 0.125 * zc
    return qult, z50


# ============================================================
# BUILD MODEL FROM THE TCL LOGIC
# ============================================================
def build_bnwf_model(params: Dict[str, float]) -> Dict[str, int]:
    ops.wipe()

    L1 = params["L1"]
    L2 = params["L2"]
    diameter = params["diameter"]
    nElePile = int(params["nElePile"])
    eleSize = (L1 + L2) / nElePile
    nNodePile = 1 + nElePile
    gamma = params["gamma"]
    phi = params["phi"]
    Gsoil = params["Gsoil"]
    puSwitch = int(params["puSwitch"])
    kSwitch = int(params["kSwitch"])
    gwtSwitch = int(params["gwtSwitch"])

    # Spring nodes
    ops.model("Basic", "-ndm", 3, "-ndf", 3)
    count = 0
    for i in range(1, nNodePile + 1):
        zCoord = eleSize * (i - 1)
        if zCoord <= L2 + 1e-12:
            ops.node(i, 0.0, 0.0, zCoord)
            ops.node(i + 100, 0.0, 0.0, zCoord)
            count += 1
    nNodeEmbed = count

    for i in range(1, nNodeEmbed + 1):
        ops.fix(i, 1, 1, 1)
        ops.fix(i + 100, 0, 1, 1)

    # Materials from Tcl procedures
    for i in range(1, nNodeEmbed + 1):
        pyDepth = L2 - eleSize * (i - 1)
        pult, y50 = get_py_param(pyDepth, gamma, phi, diameter, eleSize, puSwitch, kSwitch, gwtSwitch)
        ops.uniaxialMaterial("PySimple1", i, 2, pult, y50, params["pult_cd"])

    for i in range(2, nNodeEmbed + 1):
        pyDepth = eleSize * (i - 1)
        sigV = gamma * pyDepth
        tult, z50 = get_tz_param(phi, diameter, sigV, eleSize)
        ops.uniaxialMaterial("TzSimple1", i + 100, 2, tult, z50, params["tz_cd"])

    sigVq = gamma * L2
    qult, z50q = get_qz_param(phi, diameter, sigVq, Gsoil)
    ops.uniaxialMaterial("QzSimple1", 101, 2, qult, z50q, params["qz_suction"], params["qz_cd"])

    # zero-length spring elements
    ops.element("zeroLength", 1001, 1, 101, "-mat", 1, 101, "-dir", 1, 3)
    for i in range(2, nNodeEmbed + 1):
        ops.element("zeroLength", i + 1000, i, i + 100, "-mat", i, i + 100, "-dir", 1, 3)

    # pile nodes
    ops.model("Basic", "-ndm", 3, "-ndf", 6)
    for i in range(1, nNodePile + 1):
        zCoord = eleSize * (i - 1)
        ops.node(i + 200, 0.0, 0.0, zCoord)

    ops.geomTransf("Linear", 1, 0.0, -1.0, 0.0)

    # exact Tcl pile restraints
    ops.fix(200 + nNodePile, 0, 1, 0, 1, 0, 1)
    for i in range(202, 200 + nNodePile):
        ops.fix(i, 0, 1, 0, 1, 0, 1)

    for i in range(1, nNodeEmbed + 1):
        ops.equalDOF(i + 200, i + 100, 1, 3)

    # elasticPileSection.tcl
    ops.section("Elastic", 1, params["E"], params["A"], params["Iz"], params["Iy"], params["G"], params["J"])
    ops.uniaxialMaterial("Elastic", 3000, params["torsion_stiffness"])
    secTag3D = 3
    ops.section("Aggregator", secTag3D, 3000, "T", "-section", 1)

    for i in range(201, 200 + nElePile + 1):
        ops.element("dispBeamColumn", i, i, i + 1, secTag3D, 3, 1)

    return {
        "eleSize": eleSize,
        "nNodePile": nNodePile,
        "nNodeEmbed": nNodeEmbed,
        "pileNodeStart": 201,
        "pileNodeEnd": 200 + nNodePile,
        "pileEleStart": 201,
        "pileEleEnd": 200 + nElePile,
        "headNode": 200 + nNodePile,
    }


# ============================================================
# ANALYSIS
# ============================================================
def run_bnwf_analysis(params: Dict[str, float], info: Dict[str, int]) -> int:
    ops.timeSeries("Path", 10, "-time", 0, 10, 20, 10000, "-values", 0, 0, 1, 1, "-factor", 1)
    ops.pattern("Plain", 10, 10)
    ops.load(info["headNode"], params["head_load_x"], 0.0, 0.0, 0.0, 0.0, 0.0)

    ops.integrator("LoadControl", params["load_increment"])
    ops.numberer("RCM")
    ops.system("SparseGeneral")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-5, 20, 1)
    ops.algorithm("Newton")
    ops.analysis("Static")

    return ops.analyze(int(params["load_steps"]))


# ============================================================
# RESULTS
# ============================================================
def get_bnwf_results(info: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pile_nodes = list(range(info["pileNodeStart"], info["pileNodeEnd"] + 1))
    pile_eles = list(range(info["pileEleStart"], info["pileEleEnd"] + 1))
    spring_nodes = list(range(1, info["nNodeEmbed"] + 1))

    node_rows = []
    for node in pile_nodes:
        x, y, z = ops.nodeCoord(node)
        d = ops.nodeDisp(node)
        node_rows.append({
            "node": node,
            "x": x,
            "y": y,
            "z": z,
            "ux": d[0],
            "uy": d[1],
            "uz": d[2],
            "rx": d[3],
            "ry": d[4],
            "rz": d[5],
        })
    node_df = pd.DataFrame(node_rows)

    reaction_rows = []
    for node in spring_nodes:
        r = ops.nodeReaction(node)
        x, y, z = ops.nodeCoord(node)
        reaction_rows.append({
            "spring_node": node,
            "z": z,
            "Rx": r[0] if len(r) > 0 else 0.0,
            "Ry": r[1] if len(r) > 1 else 0.0,
            "Rz": r[2] if len(r) > 2 else 0.0,
        })
    reaction_df = pd.DataFrame(reaction_rows)

    ele_rows = []
    for ele in pile_eles:
        f = ops.eleForce(ele)
        ni = ele
        nj = ele + 1
        zi = ops.nodeCoord(ni)[2]
        zj = ops.nodeCoord(nj)[2]
        ele_rows.append({
            "element": ele,
            "zi": zi,
            "zj": zj,
            "P_i": f[0], "Vy_i": f[1], "Vz_i": f[2], "T_i": f[3], "My_i": f[4], "Mz_i": f[5],
            "P_j": f[6], "Vy_j": f[7], "Vz_j": f[8], "T_j": f[9], "My_j": f[10], "Mz_j": f[11],
        })
    ele_df = pd.DataFrame(ele_rows)
    return node_df, reaction_df, ele_df


# ============================================================
# PLOTS
# ============================================================
def plot_deformed_shape(node_df: pd.DataFrame, L2: float):
    fig, ax = plt.subplots(figsize=(6, 8))
    z = node_df["z"].values
    ux = node_df["ux"].values
    max_abs = max(np.max(np.abs(ux)), 1e-12)
    scale = min(50.0, max(1.0, 0.15 * max(z) / max_abs))

    ax.axhspan(0.0, L2, color="#d6eaf8", alpha=0.5)
    if max(z) > L2:
        ax.axhspan(L2, max(z), color="#f5f5f5", alpha=0.7)

    ax.plot(np.zeros_like(z), z, "k--", label="Undeformed")
    ax.plot(ux * scale, z, "b-", lw=2.2, label=f"Deformed x{scale:.1f}")
    ax.invert_yaxis()
    ax.set_xlabel("Horizontal displacement (scaled)")
    ax.set_ylabel("z (m)")
    ax.set_title("Pile deformed shape")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_profile(x, z, xlabel, title):
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(x, z, lw=2)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel("z (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return fig


# ============================================================
# UI
# ============================================================
st.sidebar.header("Geometry")
L1 = st.sidebar.number_input("Above-ground length L1 (m)", value=DEFAULTS["L1"], step=0.1)
L2 = st.sidebar.number_input("Embedded length L2 (m)", value=DEFAULTS["L2"], step=0.5)
diameter = st.sidebar.number_input("Pile diameter (m)", value=DEFAULTS["diameter"], step=0.05)
nElePile = st.sidebar.number_input("Number of pile elements", value=DEFAULTS["nElePile"], min_value=4, step=2)

st.sidebar.header("Soil")
gamma = st.sidebar.number_input("Unit weight gamma (kN/m3)", value=DEFAULTS["gamma"], step=0.5)
phi = st.sidebar.number_input("Friction angle phi (deg)", value=DEFAULTS["phi"], step=1.0)
Gsoil = st.sidebar.number_input("Soil shear modulus at tip Gsoil (kPa)", value=DEFAULTS["Gsoil"], step=1000.0)
puSwitch = st.sidebar.selectbox("pult method", [1, 2], index=0, format_func=lambda x: "API" if x == 1 else "Brinch Hansen")
kSwitch = st.sidebar.selectbox("k variation", [1, 2], index=0, format_func=lambda x: "API linear" if x == 1 else "Modified API parabolic")
gwtSwitch = st.sidebar.selectbox("Groundwater switch", [1, 2], index=0, format_func=lambda x: "Above GWT" if x == 1 else "Below GWT")

st.sidebar.header("Pile Section")
E = st.sidebar.number_input("E", value=DEFAULTS["E"], step=1e6, format="%.6e")
A = st.sidebar.number_input("A", value=DEFAULTS["A"], step=0.01)
Iz = st.sidebar.number_input("Iz", value=DEFAULTS["Iz"], step=0.001)
Iy = st.sidebar.number_input("Iy", value=DEFAULTS["Iy"], step=0.001)
G = st.sidebar.number_input("G", value=DEFAULTS["G"], step=1e6, format="%.6e")
J = st.sidebar.number_input("J", value=DEFAULTS["J"], step=0.001)
torsion_stiffness = st.sidebar.number_input("Elastic torsion material", value=DEFAULTS["torsion_stiffness"], step=1e8, format="%.6e")

st.sidebar.header("Loading and Analysis")
head_load_x = st.sidebar.number_input("Head lateral load Fx (kN)", value=DEFAULTS["head_load_x"], step=100.0)
load_steps = st.sidebar.number_input("Load steps", value=DEFAULTS["load_steps"], min_value=1, step=10)
load_increment = st.sidebar.number_input("LoadControl increment", value=DEFAULTS["load_increment"], step=0.01, format="%.3f")

params = {
    "L1": float(L1),
    "L2": float(L2),
    "diameter": float(diameter),
    "nElePile": int(nElePile),
    "gamma": float(gamma),
    "phi": float(phi),
    "Gsoil": float(Gsoil),
    "puSwitch": int(puSwitch),
    "kSwitch": int(kSwitch),
    "gwtSwitch": int(gwtSwitch),
    "head_load_x": float(head_load_x),
    "load_steps": int(load_steps),
    "load_increment": float(load_increment),
    "pult_cd": DEFAULTS["pult_cd"],
    "tz_cd": DEFAULTS["tz_cd"],
    "qz_suction": DEFAULTS["qz_suction"],
    "qz_cd": DEFAULTS["qz_cd"],
    "E": float(E),
    "A": float(A),
    "Iz": float(Iz),
    "Iy": float(Iy),
    "G": float(G),
    "J": float(J),
    "torsion_stiffness": float(torsion_stiffness),
}

st.subheader("Model basis")
st.write(
    "This app follows the uploaded OpenSees BNWF example: spring nodes and pile nodes are built separately, "
    "p-y, t-z, and q-z materials are generated from the translated Tcl procedures, the pile uses a 3D dispBeamColumn model, "
    "and the same fixed/equalDOF relationships are applied."
)

summary_df = pd.DataFrame([
    {"Parameter": "L1", "Value": params["L1"]},
    {"Parameter": "L2", "Value": params["L2"]},
    {"Parameter": "diameter", "Value": params["diameter"]},
    {"Parameter": "nElePile", "Value": params["nElePile"]},
    {"Parameter": "gamma", "Value": params["gamma"]},
    {"Parameter": "phi", "Value": params["phi"]},
    {"Parameter": "Gsoil", "Value": params["Gsoil"]},
    {"Parameter": "head_load_x", "Value": params["head_load_x"]},
])
st.dataframe(summary_df, use_container_width=True)

run_clicked = st.button("Run BNWF analysis", type="primary")

if not OPENSEES_AVAILABLE:
    st.warning("OpenSeesPy is not installed in this environment.")

if run_clicked and OPENSEES_AVAILABLE:
    try:
        info = build_bnwf_model(params)
        ok = run_bnwf_analysis(params, info)
        if ok != 0:
            st.error("Analysis did not converge.")
        else:
            ops.reactions()
            node_df, reaction_df, ele_df = get_bnwf_results(info)

            top = node_df.iloc[-1]
            c1, c2, c3 = st.columns(3)
            c1.metric("Top ux", f"{top['ux']:.6f} m")
            c2.metric("Max |Mz|", f"{ele_df[['Mz_i','Mz_j']].abs().to_numpy().max():.3f} kN.m")
            c3.metric("Max |soil Rx|", f"{reaction_df['Rx'].abs().max():.3f} kN")

            st.pyplot(plot_deformed_shape(node_df, params["L2"]))

            colp1, colp2 = st.columns(2)
            with colp1:
                st.pyplot(plot_profile(node_df["ux"].values, node_df["z"].values, "ux (m)", "Lateral displacement"))
                st.pyplot(plot_profile(reaction_df["Rx"].values, reaction_df["z"].values, "Spring reaction Rx (kN)", "p-y spring reaction"))
            with colp2:
                z_ele = ele_df[["zi", "zj"]].mean(axis=1).values
                mz = 0.5 * (ele_df["Mz_i"].values - ele_df["Mz_j"].values)
                vy = 0.5 * (ele_df["Vy_i"].values - ele_df["Vy_j"].values)
                st.pyplot(plot_profile(mz, z_ele, "Mz (kN.m)", "Bending moment"))
                st.pyplot(plot_profile(vy, z_ele, "Vy (kN)", "Shear"))

            st.subheader("Pile node displacements")
            st.dataframe(node_df, use_container_width=True)
            st.subheader("Spring reactions")
            st.dataframe(reaction_df, use_container_width=True)
            st.subheader("Pile element forces")
            st.dataframe(ele_df, use_container_width=True)

    except Exception as exc:
        st.exception(exc)
