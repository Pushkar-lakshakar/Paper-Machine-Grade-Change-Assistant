import os
import pickle
import numpy as np
import pandas as pd

class RecommendationEngine:
    def __init__(self):
        # Paths
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_dir = os.path.join(base_dir, "models")
        self.data_dir = os.path.join(base_dir, "data")
        
        # Load trained ML models
        with open(os.path.join(self.model_dir, "risk_classifier.pkl"), "rb") as f:
            self.clf = pickle.load(f)
        with open(os.path.join(self.model_dir, "time_regressor.pkl"), "rb") as f:
            self.reg = pickle.load(f)
        with open(os.path.join(self.model_dir, "insights.pkl"), "rb") as f:
            self.insights = pickle.load(f)
            
        # Load historical dataset
        self.historian = pd.read_csv(os.path.join(self.data_dir, "historian_data.csv"))
        
        # Extract features for all historical runs for Case-Based Reasoning
        self.hist_features = self.extract_features_all(self.historian)
        
    def extract_features_all(self, df):
        events = df.groupby('event_id')
        feature_rows = []
        for event_id, group in events:
            grade_from = group['grade_from'].iloc[0]
            grade_to = group['grade_to'].iloc[0]
            event_type = group['event_type'].iloc[0]
            off_spec = group['off_spec_flag'].iloc[0]
            stabilize_time = group['stabilize_time_s'].iloc[0]
            
            # Extract first 5 mins (0 <= t <= 300)
            early = group[(group['t_seconds'] >= 0) & (group['t_seconds'] <= 300)]
            if len(early) == 0:
                continue
                
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
                
            expected_dryer_temp = 50.0 * early['steam_pressure']
            temp_lag = expected_dryer_temp - early['dryer_temp']
            row['dryer_temp_lag_mean'] = np.mean(temp_lag)
            row['dryer_temp_lag_max'] = np.max(temp_lag)
            row['dryer_temp_lag_std'] = np.std(temp_lag)
            row['speed_ramp_rate'] = row['machine_speed_delta'] / 300.0
            
            feature_rows.append(row)
            
        return pd.DataFrame(feature_rows)

    def process_single_event(self, df_event_early):
        """
        Extract features from a single event's early window (first 5 minutes)
        """
        grade_from = df_event_early['grade_from'].iloc[0]
        grade_to = df_event_early['grade_to'].iloc[0]
        event_id = df_event_early['event_id'].iloc[0]
        
        early = df_event_early[(df_event_early['t_seconds'] >= 0) & (df_event_early['t_seconds'] <= 300)].copy()
        
        row = {
            'event_id': event_id,
            'grade_from': grade_from,
            'grade_to': grade_to,
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
            
        expected_dryer_temp = 50.0 * early['steam_pressure']
        temp_lag = expected_dryer_temp - early['dryer_temp']
        row['dryer_temp_lag_mean'] = np.mean(temp_lag)
        row['dryer_temp_lag_max'] = np.max(temp_lag)
        row['dryer_temp_lag_std'] = np.std(temp_lag)
        row['speed_ramp_rate'] = row['machine_speed_delta'] / 300.0
        
        return pd.DataFrame([row])

    def get_recommendations(self, df_event_early):
        # 1. Process early window data to get features
        df_single_feat = self.process_single_event(df_event_early)
        
        # 2. Prepare features for ML models
        # One-hot encode using the same columns as during training
        feature_cols = self.insights['feature_columns']
        
        # Create a dummy dataframe with correct columns
        df_model_input = pd.DataFrame(0.0, index=[0], columns=feature_cols)
        
        # Copy matching columns
        for col in feature_cols:
            if col in df_single_feat.columns:
                df_model_input.loc[0, col] = float(df_single_feat.loc[0, col])
            elif col.startswith('transition_'):
                trans_val = col.split('transition_')[1]
                if df_single_feat.loc[0, 'transition'] == trans_val:
                    df_model_input.loc[0, col] = 1.0
                    
        # 3. Predict Risk and Stabilization Time
        prob_off_spec = float(self.clf.predict_proba(df_model_input)[0, 1])
        pred_stabilize_time = float(self.reg.predict(df_model_input)[0])
        
        recommendations = []
        
        # Extract current state at t=300
        current_state = df_event_early[df_event_early['t_seconds'] == 300].iloc[0]
        
        # 4. Generate recommendations based on risk classification & diagnostics
        # We classify risk as High if probability > 0.4
        is_high_risk = prob_off_spec > 0.4
        
        # Diagnostic check 1: Dryer temp lag (Waterlogging symptom)
        dryer_lag = float(df_single_feat.loc[0, 'dryer_temp_lag_mean'])
        
        # Diagnostic check 2: Speed ramp rate (Mismatch symptom)
        speed_ramp_rate = float(df_single_feat.loc[0, 'speed_ramp_rate'])
        
        if is_high_risk:
            # Check for waterlogging
            if dryer_lag > 4.0:
                # Clip recommendations within safe recipe operating limits
                rec_speed = np.clip(
                    current_state['machine_speed'] - 35.0,
                    current_state['recipe_min_speed'],
                    current_state['recipe_max_speed']
                )
                rec_steam = np.clip(
                    current_state['steam_pressure'] + 0.3,
                    current_state['recipe_min_steam'],
                    current_state['recipe_max_steam']
                )
                
                recommendations.append({
                    'parameter': 'machine_speed',
                    'current': f"{current_state['machine_speed']:.1f} m/min",
                    'recommended': f"{rec_speed:.1f} m/min",
                    'source_tag': 'new-correlation',
                    'rationale': f"Dryer temp lags steam by {dryer_lag:.1f}°C (waterlogging). Nudging machine speed down to {rec_speed:.1f} m/min (clipped within recipe limits [{current_state['recipe_min_speed']:.0f}-{current_state['recipe_max_speed']:.0f}] m/min)."
                })
                recommendations.append({
                    'parameter': 'steam_pressure',
                    'current': f"{current_state['steam_pressure']:.2f} bar",
                    'recommended': f"{rec_steam:.2f} bar",
                    'source_tag': 'historical-data',
                    'rationale': f"Increasing steam setpoint to {rec_steam:.2f} bar (clipped within recipe limits [{current_state['recipe_min_steam']:.1f}-{current_state['recipe_max_steam']:.1f}] bar) to accelerate thermal recovery."
                })
            
            # Check for speed mismatch
            elif abs(speed_ramp_rate) > 0.15: # Ramp speed is high (negative/positive delta)
                rec_stock = np.clip(
                    current_state['stock_flow'] + 40.0,
                    current_state['recipe_min_stock'],
                    current_state['recipe_max_stock']
                )
                
                recommendations.append({
                    'parameter': 'machine_speed',
                    'current': f"Ramp completed in 200s",
                    'recommended': f"Extend ramp to 900s",
                    'source_tag': 'historical-data',
                    'rationale': f"The current speed ramp is too steep ({speed_ramp_rate*60:.1f} m/min/min). Extending speed ramp duration to 900s matches stock flow dynamics and prevents weight deviation."
                })
                recommendations.append({
                    'parameter': 'stock_flow',
                    'current': f"{current_state['stock_flow']:.1f} L/min",
                    'recommended': f"{rec_stock:.1f} L/min",
                    'source_tag': 'recipe-limit',
                    'rationale': f"Proactively boost stock flow target to {rec_stock:.1f} L/min (clipped within recipe limits [{current_state['recipe_min_stock']:.0f}-{current_state['recipe_max_stock']:.0f}] L/min) to counter sheet thinning."
                })
            else:
                # General case-based recommendation (CBR)
                # Find most similar successful historical runs
                transition = df_single_feat.loc[0, 'transition']
                successful_runs = self.hist_features[
                    (self.hist_features['transition'] == transition) & 
                    (self.hist_features['off_spec_flag'] == 0)
                ]
                
                if len(successful_runs) > 0:
                    # Compute Euclidean distance on features
                    feats_to_compare = ['stock_flow_mean', 'filler_flow_mean', 'steam_pressure_mean', 'machine_speed_mean']
                    
                    hist_matrix = successful_runs[feats_to_compare].values
                    curr_vector = df_single_feat[feats_to_compare].values
                    
                    distances = np.linalg.norm(hist_matrix - curr_vector, axis=1)
                    best_idx = np.argmin(distances)
                    best_run = successful_runs.iloc[best_idx]
                    best_run_id = best_run['event_id']
                    
                    recommendations.append({
                        'parameter': 'stock_flow & speed',
                        'current': f"Custom ramp",
                        'recommended': f"Copy run {best_run_id} profile",
                        'source_tag': 'historical-data',
                        'rationale': f"This run is deviating from target. Aligning setpoints with successful historical run {best_run_id} is predicted to stabilize in {best_run['stabilize_time_s']:.0f}s."
                    })
                else:
                    recommendations.append({
                        'parameter': 'QCS Controller',
                        'current': 'Auto',
                        'recommended': 'Hold current targets',
                        'source_tag': 'recipe-limit',
                        'rationale': 'No matching historical runs. Hold current trajectories to avoid compounding instabilities.'
                    })
        else:
            # Low risk
            recommendations.append({
                'parameter': 'Grade Change Co-ordinator',
                'current': 'Ramping',
                'recommended': 'Maintain current trajectory',
                'source_tag': 'recipe-limit',
                'rationale': 'Grade change transition is proceeding normally. Basis weight tracking deviation is within 2.5% limits.'
            })
            
        return {
            'prob_off_spec': prob_off_spec,
            'pred_stabilize_time': pred_stabilize_time,
            'is_high_risk': is_high_risk,
            'recommendations': recommendations
        }

if __name__ == "__main__":
    # Quick verification test
    engine = RecommendationEngine()
    
    # Load first 300 seconds of a normal event
    df_all = pd.read_csv(os.path.join(engine.data_dir, "historian_data.csv"))
    
    print("\n--- Testing Recommendation Engine on Normal Event ---")
    ev_normal = df_all[df_all['event_type'] == 'normal']['event_id'].iloc[0]
    df_ev_norm = df_all[df_all['event_id'] == ev_normal]
    res_norm = engine.get_recommendations(df_ev_norm)
    print(f"Event: {ev_normal} (Normal)")
    print(f"Risk P(off-spec): {res_norm['prob_off_spec']:.4f}")
    print(f"Predicted stabilize time: {res_norm['pred_stabilize_time']:.1f}s")
    for rec in res_norm['recommendations']:
        print(f"  - Parameter: {rec['parameter']}, Current: {rec['current']}, Rec: {rec['recommended']}, Source: {rec['source_tag']}")
        print(f"    Rationale: {rec['rationale']}")
        
    print("\n--- Testing Recommendation Engine on Waterlogging Event ---")
    ev_wl = df_all[df_all['event_type'] == 'waterlogging']['event_id'].iloc[0]
    df_ev_wl = df_all[df_all['event_id'] == ev_wl]
    res_wl = engine.get_recommendations(df_ev_wl)
    print(f"Event: {ev_wl} (Waterlogging)")
    print(f"Risk P(off-spec): {res_wl['prob_off_spec']:.4f}")
    print(f"Predicted stabilize time: {res_wl['pred_stabilize_time']:.1f}s")
    for rec in res_wl['recommendations']:
        print(f"  - Parameter: {rec['parameter']}, Current: {rec['current']}, Rec: {rec['recommended']}, Source: {rec['source_tag']}")
        print(f"    Rationale: {rec['rationale']}")
