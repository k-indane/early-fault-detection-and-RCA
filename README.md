# Live Monitor for Early Fault Detection and Root Cause Analysis

This project implements an end-to-end early fault detection and root cause analysis system for industrial process data, producing a real-time streaming monitoring service with a backend hybrid ML pipeline and a live frontend dashboard. This system detects known and unseen faults, escalates alerts, and performs root cause analysis.

The dataset used is from the Tennessee Eastman Process, a benchmark chemical process simulation commonly used in research on fault detection and process control.

## What This Project Demonstrates
- Production style ML system architecture
- Hybrid detection using supervised (XGBoost) + unsupervised models (NN Autoencoder)
- Root Cause Analysis with interpretable ML via SHAP and partial residual analysis
- Decision logic to manage hybrid model performance
- Real-time inference & streaming
- Full stack integration (Python FastAPI backend + TypeScript frontend)

## Video

Please refer to the video below for technical modeling details, and a live demo!

 **Placeholder for Video**

A full technical report covering problem introduction, EDA, feature engineering, model design, training, and evaluation is included under 'docs/'.

## System Architecture

Data + Models → Online Monitoring Engine → State & Alerts → RCA → Backend Streaming → Frontend Dashboard

1. **Data + Models**: Process data is analyzed with supervised fault classification (XGBoost) and with unsupervised anomaly detection (NN Autoencoder), including temporal feature engineering and tuned thresholds.
2. **Online Monitoring Engine**: Processes a run incrementally as a live stream to perform real-time inference, maintain internal buffers for engineered temporal features.
3. **State & Alerts**: Converts raw model outputs into stable flags, and then system states as "NORMAL", "SUSPECT", and "ALERT" using designed arbitration rules.
4. **Root Cause Analysis**: Triggers RCA at "ALERT", highlighting feature contributions for detection with SHAP (XGBoost) and partial reconstuction error residual (NN Autoencoder).
5. **Backend Streaming**: A FastAPI session streams monitoring events (tick, alert, rca, etc.) over WebSockets, emitting only data events in JSON.
6. **Frontend Dashboard**: A Next.js + TypeScript live dashboard styled to simulate an operational UI.

## Repository Structure

```
early-fault-detection-and-RCA/
├─ monitor.yaml
│  └─ Central config for monitoring
│
├─ data/
│  └─ Tennessee Eastman Process datasets
│
├─ models/                             
│  └─ Saved model artifacts
│
├─ scripts/                             
│  ├─ run_monitor.py
│  │  └─ Batch monitoring (dev tool)
│  │
│  ├─ run_online_engine.py
│     └─ Local streaming runner (dev tool)

├─ src/
│  └─ tefault/                          Core ML monitoring
│     ├─ __init__.py
│     │
│     ├─ config.py
│     │  └─ Config loader
│     │
│     ├─ data.py
│     │  └─ Run selection, labeling, and config filtering
│     │
│     ├─ features.py
│     │  ├─ Feature engineering (lags, EWMA) for XGB
│     │  ├─ apply_consecutive_rule() for stable alerts
│     │  └─ make_ae_windows_for_run() for AE sliding windows
│     │
│     ├─ scorers/
│     │  ├─ __init__.py
│     │  ├─ xgb.py
│     │  │  └─ XGBScorer
│     │  └─ ae.py
│     │     └─ AEScorer
│     │
│     ├─ state.py
│     │  └─ Applies alert arbitration and state transitions
│     │
│     ├─ rca.py
│     │  ├─ xgb_rca_shap(): SHAP-based RCA
│     │  └─ ae_rca_residuals(): partial residual-based RCA
│     │
│     ├─ io.py
│     │  └─ Output io utilities
│     │
│     ├─ monitor.py
│     │  └─ Offline orchestration and validation layer
│     │
│     └─ online_engine.py
│        └─ OnlineMonitorEngine: processes one sample at a time and emits tick/alert/rca/done events.
│
├─ apps/                                
│  ├─ backend/
│  │  ├─ __init__.py
│  │  ├─ main.py
│  │  │  └─ Backend service that powers the live dashboard by streaming in real time.
│  │  ├─ session_manager.py
│  │  │  └─ Caches artifacts/data, manages sessions, builds OnlineMonitorEngine per session
│  │
│  └─ frontend/
│     ├─ app/
│     │  ├─ layout.tsx
│     │  │  └─ Root layout wrapper
│     │  ├─ page.tsx
│     │  │  └─ Main dashboard
│     │  └─ globals.css
│     │     └─ Global CSS + Tailwind directives
│     ├─ components/
│     │ 
│     ├─ hooks/
│     │ 
│     ├─ next-env.d.ts
│     │ 
│     ├─ next.config.js
│     │ 
│     ├─ package.json
│     │  └─ Frontend dependencies
│     │ 
│     ├─ package-lock.json
│     │  └─ Lockfile for reproducible installs
│     │ 
│     ├─ tsconfig.json
│     │  └─ TypeScript compiler settings
│     │ 
│     ├─ tailwind.config.js
│     │ 
│     └─ postcss.config.js
│
├─ docs/
│  └─ Early Fault Detection and Root Cause Analysis for Manufacturing Systems with A Hybrid Machine Learning Framework.pdf
│     └─ Full project report
│
└─ README.md
```

## Installation and Setup

This guide will help you set up the project on your local machine.

### Prerequisites

Make sure you have the following installed:

- **Python 3.10 or higher** 
- **Node.js and npm** - Node.js 18+ recommended
- **Conda** - optional but recommended
- **Git** - To clone the repo

### Step 1: Clone the Repo

```bash
git clone <repository-url>
cd early-fault-detection-and-RCA
```

### Step 2: Set Up Python Environment

#### Using Conda (Recommended)

1. Create a new conda environment:
   ```bash
   conda create -n tefault python=3.10
   conda activate tefault
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the `tefault` package:
   ```bash
   pip install -e .
   ```
   
   This installs the local `tefault` package so Python can import it from anywhere in the project.

### Step 3: Set Up Frontend Dependencies

1. Navigate to the frontend directory:
   ```bash
   cd apps/frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```
   
   This will read `package.json` and install all required packages based on the versions specified in `package-lock.json`.

3. Return to the project root:
   ```bash
   cd ../..
   ```

### Step 5: Running the Services

The project requires two simutaneous services: the backend API and the frontend web app.

#### Running the Backend

1. Activate tefault environment

2. Start the backend server:
   ```bash
   uvicorn apps.backend.main:app --reload --host 0.0.0.0 --port 8000
   ```

   The backend will be available at:
   - **API**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

   **Note**: On Windows, you can just use the batch file:
   ```bash
   BEservice.bat
   ```

#### Running the Frontend

1. Open a new terminal window

2. Navigate to the frontend directory:
   ```bash
   cd apps/frontend
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

   The frontend will be available at:
   - **Web Application**: http://localhost:3000

   **Note**: On Windows, you can just use the batch file:
   ```bash
   FEservice.bat
   ```

### Step 6: Using the Application

1. Open your web browser and navigate to **http://localhost:3000**

2. Select a run from the dropdown menu

3. Click "Start Monitoring" to begin simulating real-time fault detection

4. Watch the dashboard update with:
   - Real-time charts showing XGBoost confidence and Autoencoder reconstruction error
   - Alerts when faults are detected
   - Root Cause Analysis results with feature contributions

### Data Files

Make sure the following data files are present in the `data/` directory:
- `TEP_FaultFree_Training_fault_free_training.pkl`
- `TEP_Faulty_Training_faulty_training.pkl`

# Author
Kaustubh Indane

M.S. Data Science AI - Northwestern University

Background in semiconductor process integration

# License
This project is intended for educational and portfolio purposes only.