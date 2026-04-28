import json
import math
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
st.title("Lateral Pile Analysis (LPILE-Calibrated Mode)")

DEFAULT_LAYERS = [
    {
        "name": "Sand",
        "z_top": 0.0,
        "z_bot": 20.0,
        "soilType": 2,
        "gamma": 17000.0,
        "Cd": 0.05,
        "phi_deg": 40.0,
        "k": 2.0e7,
    }
]

if "soil_layers" not in st.session_state:
    st.session_state.soil_layers = json.loads(json.dumps(DEFAULT_LAYERS))

# ============================================================
# CORE FIXED p-y IMPLEMENTATION (LPILE-aligned)
# ============================================================

def get_layer(z, layers):
    for l in layers:
        if l["z_top"] <= z < l["z_bot"]:
            return l
    return layers[-1]


def tributary_length(i, n, dz):
    return 0.5 * dz if i in [1, n] else dz


def derive_py_params(layer, z, D, tlen):
    soil = layer["soilType"]
    z = max(z, 1e-6)

    if soil == 1:
        c = layer["c"]
        eps50 = layer.get("eps50", 0.02)
        pu = 9 * c * D
        y50 = 2.5 * eps50 * D

    elif soil == 2:
        phi = np.radians(layer["phi_deg"])
        gamma = layer["gamma"]
        k = layer["k"]  # ✅ USE USER INPUT DIRECTLY

        # API ultimate resistance (simplified but correct form)
        pu = gamma * z * D * np.tan(phi)

        # ✅ CRITICAL FIX: enforce k*z stiffness
        y50 = pu / (k * z)

    else:
        raise ValueError("Invalid soilType")

    pu = max(pu, 1.0)
    y50 = max(y50, 1e-6)

    return pu * tlen, y50


# ============================================================
# MODEL
# ============================================================

def build_model(p, layers):
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    n = p["N_ELE"] + 1
    dz = p["PILE_LENGTH"] / p["N_ELE"]
    A = math.pi * p["PILE_DIAMETER"]**2 / 4
    I = math.pi * p["PILE_DIAMETER"]**4 / 64

    for i in range(n):
        ops.node(i+1, 0, -i*dz)

    if p["BASE_CONDITION"] == "fixed":
        ops.fix(n,1,1,1)
    else:
        ops.fix(n,1,1,0)

    if p["HEAD_CONDITION"] == "fixed":
        ops.fix(1,0,0,1)
    else:
        ops.fix(1,0,0,0)

    ops.geomTransf("Linear",1)

    for e in range(1,n):
        ops.element("elasticBeamColumn", e,e,e+1,A,p["E_PILE"],I,1)

    mat = 1000
    ele = 2000

    for i in range(1,n):
        z = (i-1)*dz
        layer = get_layer(z,layers)
        tlen = tributary_length(i,n,dz)
        pult,y50 = derive_py_params(layer,z,p["PILE_DIAMETER"],tlen)

        ops.node(10000+i,0,-z)
        ops.fix(10000+i,1,1,1)

        mat +=1
        ops.uniaxialMaterial("PySimple1",mat,layer["soilType"],pult,y50,layer.get("Cd",0.05),0.0)

        ele +=1
        ops.element("zeroLength",ele,i,10000+i,"-mat",mat,"-dir",1)


def run_analysis(p):
    ops.timeSeries("Linear",1)
    ops.pattern("Plain",1,1)
    ops.load(1,p["H"],-p["P"],p["M"])

    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormDispIncr",1e-8,100)
    ops.algorithm("Newton")
    ops.integrator("LoadControl",0.05)
    ops.analysis("Static")

    if ops.analyze(20)!=0:
        ops.algorithm("ModifiedNewton")
        ops.analyze(40)


def get_disp(n):
    z=[]; ux=[]
    for i in range(1,n+1):
        _,y=ops.nodeCoord(i)
        z.append(-y)
        ux.append(ops.nodeDisp(i)[0])
    return np.array(z),np.array(ux)


def plot(z,ux,layers):
    fig,ax=plt.subplots(figsize=(6,8))

    for l in layers:
        ax.axhspan(l["z_top"],l["z_bot"],alpha=0.2)

    scale = max(1,0.2*max(z)/max(abs(ux)+1e-9))
    ax.plot(np.zeros_like(z),z,'k--')
    ax.plot(ux*scale,z,'b')

    ax.invert_yaxis()
    ax.set_title("Pile deformation")
    ax.grid()
    return fig

# ============================================================
# UI
# ============================================================

st.sidebar.header("Inputs")
H = st.sidebar.number_input("H (N)",value=1e4)
P = st.sidebar.number_input("P (N)",value=0.0)
M = st.sidebar.number_input("M (Nm)",value=0.0)
L = st.sidebar.number_input("Pile length",value=30.0)
N = st.sidebar.number_input("Elements",value=48)
D = st.sidebar.number_input("Diameter",value=0.8)
E = st.sidebar.number_input("E",value=3e10)

HEAD = st.sidebar.selectbox("Head",["free","fixed"])
BASE = st.sidebar.selectbox("Base",["pinned","fixed"])

layers = st.session_state.soil_layers
st.subheader("Soil Layers")
st.dataframe(pd.DataFrame(layers))

if st.button("Run analysis") and OPENSEES_AVAILABLE:

    params = {
        "H":H,"P":P,"M":M,
        "PILE_LENGTH":L,
        "N_ELE":int(N),
        "PILE_DIAMETER":D,
        "E_PILE":E,
        "HEAD_CONDITION":HEAD,
        "BASE_CONDITION":BASE
    }

    build_model(params,layers)
    run_analysis(params)
    z,ux = get_disp(int(N)+1)

    st.pyplot(plot(z,ux,layers))

    st.write("Top displacement (mm):", ux[0]*1000)

elif not OPENSEES_AVAILABLE:
    st.warning("Install OpenSeesPy")
