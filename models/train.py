import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

def load_data():
    csv_path = os.path.join(DATA_DIR, "historian_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Historian data not found at {csv_path}. Run generator.py first.")
    return pd.read_csv(csv_path)

def extract_features(df):
    print("Extracting early-window features (first 5 minutes of each event)...")
    events = df.groupby('event_id')
    feature_rows = []
    
    for event_id, group in events:
        # Metadata
        grade_from = group['grade_from'].iloc[0]
        grade_to = group['grade_to'].iloc[0]
        event_type = group['event_type'].iloc[0]
        off_spec = group['off_spec_flag'].iloc[0]
        stabilize_time = group['stabilize_time_s'].iloc[0]
        
        # Early window: 0 <= t <= 300s
        early = group[(group['t_seconds'] >= 0) & (group['t_seconds'] <= 300)].copy()
        
        # Calculate features
        row = {
            'event_id': event_id,
            'grade_from': grade_from,
            'grade_to': grade_to,
            'event_type': event_type,
            'off_spec_flag': off_spec,
            'stabilize_time_s': stabilize_time,
            'transition': f"{grade_from}_{grade_to}"
        }
        
        variables = ['stock_flow', 'filler_flow', 'steam_pressure', 'machine_speed', 'moisture', 'ash', 'caliper', 'basis_weight', 'deviation', 'dryer_temp']
        for var in variables:
            vals = early[var].values
            row[f'{var}_mean'] = np.mean(vals)
            row[f'{var}_std'] = np.std(vals)
            row[f'{var}_min'] = np.min(vals)
            row[f'{var}_max'] = np.max(vals)
            row[f'{var}_delta'] = vals[-1] - vals[0]
            
        # 1. Dryer temp lag proxy: mean of (50 * steam_pressure - dryer_temp)
        expected_dryer_temp = 50.0 * early['steam_pressure']
        temp_lag = expected_dryer_temp - early['dryer_temp']
        row['dryer_temp_lag_mean'] = np.mean(temp_lag)
        row['dryer_temp_lag_max'] = np.max(temp_lag)
        row['dryer_temp_lag_std'] = np.std(temp_lag)
        
        # 2. Speed ramp rate proxy
        row['speed_ramp_rate'] = row['machine_speed_delta'] / 300.0
        
        feature_rows.append(row)
        
    df_features = pd.DataFrame(feature_rows)
    return df_features

def run_correlation_mining(df_all, df_features):
    print("Mining correlations and cross-loop relationships...")
    
    # 1. Over all events, compute correlation matrix of raw signals during transitions (t >= 0)
    df_trans = df_all[df_all['t_seconds'] >= 0].copy()
    signals = ['stock_flow', 'filler_flow', 'steam_pressure', 'machine_speed', 'moisture', 'ash', 'caliper', 'basis_weight', 'deviation', 'dryer_temp']
    corr_matrix = df_trans[signals].corr()
    
    # 2. Specifically look at correlations with the basis weight deviation
    bw_corr = corr_matrix['deviation'].sort_values(ascending=False)
    
    # 3. Mined lagged cross-correlations between dryer_temp and moisture
    # Let's pick a waterlogged run to calculate lagged correlations
    wl_event = df_all[df_all['event_type'] == 'waterlogging']['event_id'].iloc[0]
    df_wl = df_all[df_all['event_id'] == wl_event].copy()
    
    lags = np.arange(-10, 11)  # Lags in steps of 10s (-100s to +100s)
    cross_corrs = []
    
    for lag in lags:
        if lag < 0:
            wl_moist = df_wl['moisture'].iloc[-lag:].values
            wl_temp = df_wl['dryer_temp'].iloc[:lag].values
        elif lag > 0:
            wl_moist = df_wl['moisture'].iloc[:-lag].values
            wl_temp = df_wl['dryer_temp'].iloc[lag:].values
        else:
            wl_moist = df_wl['moisture'].values
            wl_temp = df_wl['dryer_temp'].values
            
        cross_corrs.append(np.corrcoef(wl_temp, wl_moist)[0, 1])
        
    lagged_corr_data = {
        'lags_seconds': lags * 10,
        'correlation': cross_corrs
    }
    
    # Identify the lag with the strongest correlation (negative, since dryer_temp increases and moisture decreases)
    best_lag_idx = np.argmin(cross_corrs)
    best_lag_seconds = lags[best_lag_idx] * 10
    best_corr = cross_corrs[best_lag_idx]
    
    print(f"Mined Lagged Correlation: dryer_temp leads moisture with strongest negative correlation ({best_corr:.3f}) at lag of {best_lag_seconds} seconds.")
    
    # Save correlation heatmap plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1.0, vmax=1.0)
    plt.title("Process Variables Correlation Matrix during Transitions")
    plt.tight_layout()
    corr_plot_path = os.path.join(MODELS_DIR, "correlation_matrix.png")
    plt.savefig(corr_plot_path, dpi=150)
    print(f"Correlation plot saved to: {corr_plot_path}")
    
    insights = {
        'corr_matrix': corr_matrix.to_dict(),
        'bw_deviation_correlation': bw_corr.to_dict(),
        'lagged_correlation': lagged_corr_data,
        'best_lag_seconds': best_lag_seconds,
        'best_corr': best_corr
    }
    
    return insights

def train_models(df_features, insights):
    print("Preparing training datasets...")
    
    # One-hot encode the categorical transition
    df_model = pd.get_dummies(df_features, columns=['transition'], drop_first=True)
    
    # Define features and targets
    exclude_cols = ['event_id', 'grade_from', 'grade_to', 'event_type', 'off_spec_flag', 'stabilize_time_s']
    feature_cols = [c for c in df_model.columns if c not in exclude_cols]
    
    X = df_model[feature_cols].astype(float)
    y_class = df_model['off_spec_flag'].astype(int)
    y_reg = df_features['stabilize_time_s'].astype(float)
    
    # Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.25, random_state=42, stratify=y_class
    )
    
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 1. Train Risk Classifier
    print("Training predictive risk classifier (RandomForest)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(X_train, y_class_train)
    
    # Evaluate Classifier
    y_class_pred = clf.predict(X_test)
    y_class_prob = clf.predict_proba(X_test)[:, 1]
    
    print("\n--- Classifier Performance (Test Set) ---")
    print(classification_report(y_class_test, y_class_pred))
    roc_auc = roc_auc_score(y_class_test, y_class_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_class_test, y_class_pred))
    
    # 2. Train Time-to-Stabilize Regressor
    print("\nTraining stabilization time regressor (RandomForest)...")
    reg = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    reg.fit(X_train, y_reg_train)
    
    # Evaluate Regressor
    y_reg_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_reg_test, y_reg_pred)
    r2 = r2_score(y_reg_test, y_reg_pred)
    print("\n--- Regressor Performance (Test Set) ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f} seconds")
    print(f"R-squared: {r2:.4f}")
    
    # Save Feature Importance
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_imp = [{'feature': feature_cols[idx], 'importance': float(importances[idx])} for idx in indices]
    
    # Print Top 10 Features
    print("\nTop 10 features for predicting grade transition risk:")
    for i in range(min(10, len(feature_imp))):
        print(f"{i+1}. {feature_imp[i]['feature']}: {feature_imp[i]['importance']:.4f}")
        
    # Check dryer_temp features
    dryer_features = [f for f in feature_imp if 'dryer_temp' in f['feature']]
    print("\nMined Hidden Correlation Importance (Dryer Temperature Features):")
    for f in dryer_features:
        print(f"- {f['feature']}: {f['importance']:.4f}")
        
    # Save models and insights
    with open(os.path.join(MODELS_DIR, "risk_classifier.pkl"), "wb") as f:
        pickle.dump(clf, f)
    with open(os.path.join(MODELS_DIR, "time_regressor.pkl"), "wb") as f:
        pickle.dump(reg, f)
        
    # Save evaluation metrics inside insights dictionary
    cm = confusion_matrix(y_class_test, y_class_pred)
    insights['evaluation_metrics'] = {
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'classification_report': classification_report(y_class_test, y_class_pred, output_dict=True),
        'mae_seconds': float(mae),
        'r2_score': float(r2)
    }

    insights['feature_importance'] = feature_imp
    insights['feature_columns'] = feature_cols
    
    with open(os.path.join(MODELS_DIR, "insights.pkl"), "wb") as f:
        pickle.dump(insights, f)
        
    print("\nModels and insights successfully trained and saved!")

def main():
    df_all = load_data()
    df_features = extract_features(df_all)
    insights = run_correlation_mining(df_all, df_features)
    train_models(df_features, insights)

if __name__ == "__main__":
    main()
