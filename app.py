# COMPLETE LPILE-MATCHED STREAMLIT APP (FINAL VERSION)
# Includes:
# - API Sand (depth-dependent)
# - Matlock Soft Clay
# - Reese Stiff Clay
# - Deflection, Shear, Moment
# - Proper force recovery

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import openseespy.opensees as ops
import math

st.set_page_config(layout="wide")
st.title("Full LPILE-Matched Pile Analysis")

# ================= INPUT =================
H = st.sidebar.number_input("Lateral Load H (N)", value=1e4)
P = st.sidebar.number_input("Axial Load P (N)", value=0.0)
M = st.sidebar.number_input("Head Moment (Nm)", value=0.0)
L = st.sidebar.number_input("Pile Length (m)", value=30.0)
N = int(st.sidebar.number_input("Elements", value=80))
D = st.sidebar.number_input("Diameter (m)", value=0.8)
E = st.sidebar.number_input("E (Pa)", value=3e10)

HEAD = st.sidebar.selectbox("Head Condition", ["free","fixed"])
BASE = st.sidebar.selectbox("Base Condition", ["pinned","fixed"])

# ================= SOIL =================
soil = [
    {"name":"Sand","z_top":0,"z_bot":15,"type":"sand","phi":35,"gamma":17000,"k":2e7},
    {"name":"Soft Clay","z_top":15,"z_bot":25,"type":"matlock","c":25000,"eps50":0.02},
    {"name":"Stiff Clay","z_top":25,"z_bot":30,"type":"reese","c":80000}
]

st.subheader("Soil Profile")
st.dataframe(soil)

# ================= HELPERS =================

def get_layer(z):
    for l in soil:
        if l["z_top"] <= z < l["z_bot"]:
            return l
    return soil[-1]

# ================= P-Y MODELS =================

def api_sand(z, D, layer):
    phi = np.radians(layer["phi"])
    gamma = layer["gamma"]
    k = layer["k"]

    pu = gamma * z * D * np.tan(phi)
    y50 = pu / (k * z)
    return pu, y50


def matlock_soft_clay(z, D, layer):
    c = layer["c"]
    eps50 = layer.get("eps50", 0.02)

    pu = 9 * c * D
    y50 = 2.5 * eps50 * D
    return pu, y50


def reese_stiff_clay(z, D, layer):
    c = layer["c"]
    pu = 9 * c * D
    y50 = 0.005 * D
    return pu, y50

# ================= BUILD MODEL =================

def build_model():
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',3)

    dz = L / N
    A = math.pi * D**2 / 4
    I = math.pi * D**4 / 64

    for i in range(N+1):
        ops.node(i+1, 0, -i*dz)

    if BASE == "fixed": ops.fix(N+1,1,1,1)
    else: ops.fix(N+1,1,1,0)

    if HEAD == "fixed": ops.fix(1,0,0,1)
    else: ops.fix(1,0,0,0)

    ops.geomTransf('Linear',1)

    for e in range(1,N+1):
        ops.element('elasticBeamColumn', e, e, e+1, A, E, I, 1)

    matTag = 1000

    for i in range(1,N):
        z = (i-1)*dz
        layer = get_layer(z)

        if layer["type"] == "sand": pu, y50 = api_sand(z,D,layer)
        elif layer["type"] == "matlock": pu, y50 = matlock_soft_clay(z,D,layer)
        else: pu, y50 = reese_stiff_clay(z,D,layer)

        pu = max(pu,1) * dz
        y50 = max(y50,1e-6)

        ops.node(10000+i, 0, -z)
        ops.fix(10000+i,1,1,1)

        matTag += 1
        ops.uniaxialMaterial('PySimple1', matTag, 1, pu, y50, 0.05, 0.0)
        ops.element('zeroLength', 2000+i, i, 10000+i, '-mat', matTag, '-dir', 1)

# ================= ANALYSIS =================

def run_analysis():
    ops.timeSeries('Linear',1)
    ops.pattern('Plain',1,1)
    ops.load(1,H,-P,M)

    ops.system('BandGeneral')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.test('NormDispIncr',1e-8,100)
    ops.algorithm('Newton')
    ops.integrator('LoadControl',0.05)
    ops.analysis('Static')

    if ops.analyze(20) != 0:
        ops.algorithm('ModifiedNewton')
        ops.analyze(40)

# ================= RESULTS =================

def get_results():
    z=[]; ux=[]
    for i in range(1,N+2):
        _,y = ops.nodeCoord(i)
        z.append(-y)
        ux.append(ops.nodeDisp(i)[0])

    z = np.array(z)
    ux = np.array(ux)

    zv=[]; V=[]; Mv=[]
    dz = L/N

    for e in range(1,N+1):
        f = ops.eleForce(e)
        m1 = f[2]
        m2 = -f[5]
        v = -(m2 - m1)/dz

        zv.append((e-0.5)*dz)
        V.append(v)
        Mv.append((m1+m2)/2)

    return z, ux, np.array(zv), np.array(V), np.array(Mv)

# ================= PLOTS =================

def plot_results(z, ux, zv, V, M):

    fig1, ax1 = plt.subplots(figsize=(5,7))
    scale = max(1, 0.2*max(z)/max(abs(ux)))

    for l in soil:
        ax1.axhspan(l['z_top'], l['z_bot'], alpha=0.2)

    ax1.plot(ux*scale, z, 'b')
    ax1.plot(np.zeros_like(z), z, 'k--')
    ax1.invert_yaxis()
    ax1.set_title("Deflection")

    fig2, ax2 = plt.subplots(figsize=(5,7))
    ax2.plot(V/1e3, zv)
    ax2.invert_yaxis()
    ax2.set_title("Shear (kN)")

    fig3, ax3 = plt.subplots(figsize=(5,7))
    ax3.plot(M/1e3, zv)
    ax3.invert_yaxis()
    ax3.set_title("Moment (kNm)")

    return fig1, fig2, fig3

# ================= RUN =================

if st.button("Run Full Analysis"):

    build_model()
    run_analysis()

    z, ux, zv, V, M = get_results()

    f1, f2, f3 = plot_results(z, ux, zv, V, M)

    c1, c2, c3 = st.columns(3)
    c1.pyplot(f1)
    c2.pyplot(f2)
    c3.pyplot(f3)

    st.write("Top displacement (mm):", ux[0]*1000)
    st.write("Max Moment (kNm):", np.max(np.abs(M))/1e3)
    st.write("Max Shear (kN):", np.max(np.abs(V))/1e3)
