import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

try:
    import openseespy.opensees as ops
    OPENSEES_AVAILABLE = True
except:
    OPENSEES_AVAILABLE = False

# ============================================================
# CLEAN BNWF APP (FIXED SYNTAX)
# ============================================================

st.set_page_config(layout="wide")
st.title("BNWF Pile Model (OpenSees)")

# ---------- DEFAULTS (FIXED) ----------
DEFAULTS = {
    "L1": 1.0,
    "L2": 20.0,
    "diameter": 1.0,
    "nElePile": 40,
    "gamma": 17.0,
    "phi": 36.0,
    "Gsoil": 150000.0,
    "head_load_x": 3500.0
}

# ---------- INPUT ----------
L2 = st.sidebar.number_input("Embedded Length", value=DEFAULTS["L2"])
D = st.sidebar.number_input("Diameter", value=DEFAULTS["diameter"])
N = int(st.sidebar.number_input("Elements", value=DEFAULTS["nElePile"]))
H = st.sidebar.number_input("Head Load", value=DEFAULTS["head_load_x"])

# ---------- SIMPLE MODEL ----------
def run_model():
    ops.wipe()
    ops.model('basic','-ndm',2,'-ndf',3)

    dz = L2/N
    E = 3e10
    A = np.pi*D**2/4
    I = np.pi*D**4/64

    # nodes
    for i in range(N+1):
        ops.node(i+1,0,-i*dz)

    ops.fix(N+1,1,1,0)

    ops.geomTransf('Linear',1)

    for e in range(1,N+1):
        ops.element('elasticBeamColumn',e,e,e+1,A,E,I,1)

    # springs
    for i in range(1,N):
        k = 1e7*(i/N)
        ops.node(100+i,0,-i*dz)
        ops.fix(100+i,1,1,1)
        ops.uniaxialMaterial('Elastic',200+i,k)
        ops.element('zeroLength',300+i,i,100+i,'-mat',200+i,'-dir',1)

    # load
    ops.timeSeries('Linear',1)
    ops.pattern('Plain',1,1)
    ops.load(1,H,0,0)

    ops.system('BandGeneral')
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.integrator('LoadControl',0.1)
    ops.analysis('Static')
    ops.analyze(10)

    z=[]; ux=[]
    for i in range(1,N+2):
        _,y=ops.nodeCoord(i)
        z.append(-y)
        ux.append(ops.nodeDisp(i)[0])

    return np.array(z),np.array(ux)

# ---------- RUN ----------
if st.button("Run"):
    if not OPENSEES_AVAILABLE:
        st.error("OpenSeesPy not installed")
    else:
        z,ux = run_model()

        fig,ax=plt.subplots()
        ax.plot(ux,z)
        ax.invert_yaxis()
        ax.set_title("Deflection")
        st.pyplot(fig)
