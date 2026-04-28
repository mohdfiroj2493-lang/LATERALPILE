# Pile Lateral Analysis Streamlit App

A Streamlit application for lateral pile analysis using **OpenSeesPy** with:

- nonlinear **p-y springs** using `PySimple1`
- **free** or **fixed** pile head
- **fixed** or **pinned** pile base
- lateral deflection, soil reaction, shear, and moment plots
- spring, nodal force, and element-end force tables
- CSV export for key results

## Repository structure

```text
pile-streamlit-app/
├── app.py
├── requirements.txt
└── README.md
```

## Features

- Sidebar controls for pile geometry, stiffness, loads, and boundary conditions
- Soil profile editable as JSON
- Clay and sand layer support:
  - `soilType = 1` for clay with `c` and optional `eps50`
  - `soilType = 2` for sand with `phi_deg` and `gamma`
- Automatic derivation of `pult` and `y50` for `PySimple1`
- Downloadable CSV files for:
  - spring properties
  - node force table
  - element end force table

## Local setup

### 1) Create a folder and copy files

Save `app.py`, `requirements.txt`, and `README.md` into one folder.

### 2) Create and activate a virtual environment

#### Windows PowerShell

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Run the app

```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select your GitHub repository.
5. Set the main file path to `app.py`.
6. Deploy.

## Create the GitHub repository

Example using Git locally:

```bash
git init
git add .
git commit -m "Initial commit: pile lateral analysis Streamlit app"
git branch -M main
git remote add origin https://github.com/<your-username>/pile-streamlit-app.git
git push -u origin main
```

## Notes

- The application preserves the engineering logic from your OpenSeesPy script and wraps it in a web UI.
- `pult` and `y50` are still derived internally from clay or sand input using the same placeholder/API-Reese style relationships from your script.
- For a pinned base, the bottom moment should trend close to zero because it is recovered from the last beam element end force.

## Recommended next improvements

- Add multi-load-case support
- Add upload/download for soil profile JSON
- Add plot image export
- Add validation for overlapping or missing soil layers
- Add unit toggle support
