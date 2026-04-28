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
    # elasticPileSection.tcl translated directly, but use 3D elasticBeamColumn in OpenSeesPy
    # This preserves the same elastic section properties from the Tcl example and avoids
    # the OpenSeesPy dispBeamColumn argument/signature mismatch seen in deployment.
    for i in range(201, 200 + nElePile + 1):
        ops.element(
            "elasticBeamColumn",
            i,
            i,
            i + 1,
            params["A"],
            params["E"],
            params["G"],
            params["J"],
            params["Iy"],
            params["Iz"],
            1,
        )

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
