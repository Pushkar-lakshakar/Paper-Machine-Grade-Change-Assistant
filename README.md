# Grade Change Intelligence (GCI) for Paper Making Process

GCI is a hackathon solution built for Honeywell's "Grade Change Intelligence in Paper Making Process" challenge. It sits on top of existing QCS MD MPC systems, mining historical historian data to predict grade change risks, recommend corrective setpoints, explain the underlying physical rationales, and record operator decisions to close the loop.

---

## Project Structure

```
./  (Project Root)
├── data/
│   ├── generator.py         # Physics-based paper machine event simulator
│   ├── historian_data.csv   # Simulated time-series dataset (200 events)
│   └── example_events.png   # Diagnostic plot showing good vs. bad transitions
├── models/
│   ├── train.py             # Feature engineering, correlation mining, and RF model training
│   ├── risk_classifier.pkl  # Saved RandomForestClassifier for off-spec prediction
│   ├── time_regressor.pkl   # Saved RandomForestRegressor for stabilization time prediction
│   ├── insights.pkl         # Saved feature importances and lag-correlation matrix
│   └── correlation_matrix.png # Diagnostic plot of mined correlations
├── recommendation/
│   └── engine.py            # Case-Based Reasoning & diagnostic recommendation API
├── dashboard/
│   ├── app.py               # Streamlit-based operator dashboard
│   └── operator_feedback.db # SQLite database storing Accept/Reject operator feedback
├── docs/
│   └── architecture.md      # Detailed GCI architecture design document
├── run_pipeline.py          # Master script to execute data generation and training sequentially
└── README.md                # This running instruction file
```

---

## Installation & Setup

1. **Pre-requisites:**
   Ensure Python 3.11+ is installed. GCI requires the following packages (which are pre-installed in this environment):
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn streamlit
   ```

2. **Database Initialization:**
   The SQLite feedback database is initialized automatically the first time you boot up the dashboard.

---

## Execution Guide

You can run the entire pipeline end-to-end using the following steps:

### Step 1: Run data generation & ML training pipeline
We have provided a master script `run_pipeline.py` that generates the simulated historian data and trains the machine learning models:
```bash
python run_pipeline.py
```
*Alternatively, you can run the modules manually:*
```bash
python data/generator.py
python models/train.py
```

### Step 2: Launch the Operator Dashboard
Launch the Streamlit web application:
```bash
streamlit run dashboard/app.py
```
This will launch a web browser tab at `http://localhost:8501/` displaying the Honeywell GCI interface.

---

## Dashboard Walkthrough & Features

- **Grade Transition Monitor:** Select from 200 different simulated grade change events and filter them by type (`normal`, `waterlogging`, `speed_mismatch`).
- **Simulated Real-Time Playback:** Slide the time control on the sidebar to see how the dashboard predicts risk, updates recommendations, and flags anomalies at different points in the grade change process.
- **Predictive Risk & Est. Stabilization:** Displays the real-time probability of going off-spec (deviating >2.5% from setpoint) and estimated seconds to reach steady-state.
- **Explainable Recommendations:** Review recommended target changes (e.g. slowing speed, boosting stock flow, increasing steam targets) complete with plain-English rationales and source tags.
- **Action Feedback Capture:** Click **Accept** or **Reject** next to recommendations. These are instantly saved into the SQLite database.
- **New Correlation Discovered Panel:** Inspect the correlation matrix and see a details panel explaining how dryer temperature lag leads to moisture spikes.
- **Analytics & Learning Loop:** View cumulative acceptance statistics and the recent feedback database table.

---

## Submission Guide

To package the deliverables for the hackathon portal:
1. Ensure `data/generator.py` and `models/train.py` have been run to create the model `.pkl` and CSV files.
2. Zip the entire project root folder. The zipped archive will contain:
   - All Python modules (`data/`, `models/`, `recommendation/`, `dashboard/`)
   - Documentation (`docs/` and `README.md`)
   - Saved models and data arrays
   - A copy of your presentation PDF (if available)
3. Upload the `.zip` file directly to the hackathon portal.
