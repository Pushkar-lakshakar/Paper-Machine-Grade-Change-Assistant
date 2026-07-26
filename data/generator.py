import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Ensure the output directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Grade definitions and setpoints
# Grade 1 (Lightweight), Grade 2 (Medium), Grade 3 (Heavy)
GRADE_SETPOINTS = {
    'G1': {'bw': 55.0, 'stock': 600.0, 'filler': 100.0, 'steam': 1.8, 'speed': 450.0, 'moisture': 5.8, 'ash': 13.3, 'caliper': 80.0, 'dryer_temp': 90.0},
    'G2': {'bw': 82.5, 'stock': 800.0, 'filler': 150.0, 'steam': 2.4, 'speed': 400.0, 'moisture': 6.1, 'ash': 15.0, 'caliper': 100.0, 'dryer_temp': 120.0},
    'G3': {'bw': 118.0, 'stock': 1000.0, 'filler': 200.0, 'steam': 3.0, 'speed': 350.0, 'moisture': 6.4, 'ash': 16.0, 'caliper': 120.0, 'dryer_temp': 150.0}
}

# Recipe limits (min/max range per controlled/monitored variable)
RECIPE_LIMITS = {
    'stock_flow': {'min': 500.0, 'max': 1200.0},
    'filler_flow': {'min': 80.0, 'max': 300.0},
    'steam_pressure': {'min': 1.5, 'max': 3.5},
    'machine_speed': {'min': 300.0, 'max': 500.0},
    'moisture': {'min': 4.5, 'max': 8.0},
    'ash': {'min': 10.0, 'max': 20.0},
    'caliper': {'min': 70.0, 'max': 140.0},
    'basis_weight': {'min': 50.0, 'max': 130.0}
}

def generate_event(event_id, grade_from, grade_to, event_type, seed):
    np.random.seed(seed)
    
    # Time parameters: -300s to 2700s, 10s steps (301 steps)
    t = np.arange(-300, 2710, 10)
    n_steps = len(t)
    
    sp_from = GRADE_SETPOINTS[grade_from]
    sp_to = GRADE_SETPOINTS[grade_to]
    
    # Ramp times
    ramp_start = 0
    ramp_duration = 900  # Default 15 minutes (900 seconds)
    
    # Speed ramp duration is much faster in case of mismatch
    speed_ramp_duration = 900
    if event_type == 'speed_mismatch':
        speed_ramp_duration = 200  # Speed changes too fast
        
    # Helper to generate ramping trajectories
    def get_ramp_trajectory(val_from, val_to, duration):
        traj = np.zeros(n_steps)
        for i, curr_t in enumerate(t):
            if curr_t < 0:
                traj[i] = val_from
            elif curr_t > duration:
                traj[i] = val_to
            else:
                traj[i] = val_from + (val_to - val_from) * (curr_t / duration)
        return traj

    # Setpoint Trajectories
    stock_flow_sp = get_ramp_trajectory(sp_from['stock'], sp_to['stock'], ramp_duration)
    filler_flow_sp = get_ramp_trajectory(sp_from['filler'], sp_to['filler'], ramp_duration)
    steam_pressure_sp = get_ramp_trajectory(sp_from['steam'], sp_to['steam'], ramp_duration)
    machine_speed_sp = get_ramp_trajectory(sp_from['speed'], sp_to['speed'], speed_ramp_duration)
    basis_weight_sp = get_ramp_trajectory(sp_from['bw'], sp_to['bw'], ramp_duration)
    
    # Process lags configuration
    dt = 10.0
    if event_type == 'normal':
        # Fast, well-tuned loops for normal transitions
        tau_stock = 8.0
        tau_filler = 8.0
        tau_steam = 8.0
        tau_speed = 8.0
        tau_dryer = 20.0
        tau_mass = 5.0
        delay_steps = 0
        tau_bw = 5.0
    else:
        # Standard un-optimized process lags
        tau_stock = 30.0
        tau_filler = 30.0
        tau_steam = 25.0
        tau_speed = 20.0
        tau_dryer = 60.0
        tau_mass = 25.0
        delay_steps = 2
        tau_bw = 15.0
        
    if event_type == 'waterlogging':
        # Dryer temp lags massively
        tau_dryer = 200.0

    # Initialize state arrays
    stock_flow = np.zeros(n_steps)
    filler_flow = np.zeros(n_steps)
    steam_pressure = np.zeros(n_steps)
    machine_speed = np.zeros(n_steps)
    dryer_temp = np.zeros(n_steps)
    moisture = np.zeros(n_steps)
    ash = np.zeros(n_steps)
    caliper = np.zeros(n_steps)
    mass_flow_eff = np.zeros(n_steps)
    basis_weight = np.zeros(n_steps)
    
    # Init values
    stock_flow[0] = sp_from['stock']
    filler_flow[0] = sp_from['filler']
    steam_pressure[0] = sp_from['steam']
    machine_speed[0] = sp_from['speed']
    dryer_temp[0] = sp_from['dryer_temp']
    moisture[0] = sp_from['moisture']
    ash[0] = sp_from['ash']
    caliper[0] = sp_from['caliper']
    mass_flow_eff[0] = sp_from['stock'] * 0.9 + sp_from['filler'] * 0.1
    basis_weight[0] = sp_from['bw']
    
    for k in range(1, n_steps):
        # 1. Stock & filler flows with realistic process noise
        alpha_stock = dt / (tau_stock + dt)
        stock_flow[k] = stock_flow[k-1] + alpha_stock * (stock_flow_sp[k] - stock_flow[k-1]) + np.random.normal(0, 1.8)
        
        alpha_filler = dt / (tau_filler + dt)
        filler_flow[k] = filler_flow[k-1] + alpha_filler * (filler_flow_sp[k] - filler_flow[k-1]) + np.random.normal(0, 0.8)
        
        # 2. Steam pressure
        alpha_steam = dt / (tau_steam + dt)
        steam_pressure[k] = steam_pressure[k-1] + alpha_steam * (steam_pressure_sp[k] - steam_pressure[k-1]) + np.random.normal(0, 0.02)
        
        # 3. Dryer temperature (depends on steam pressure)
        alpha_dryer = dt / (tau_dryer + dt)
        if event_type == 'waterlogging':
            expected_temp = sp_from['dryer_temp'] + 0.72 * (50.0 * steam_pressure[k] - sp_from['dryer_temp'])
        else:
            expected_temp = 50.0 * steam_pressure[k]
        dryer_temp[k] = dryer_temp[k-1] + alpha_dryer * (expected_temp - dryer_temp[k-1]) + np.random.normal(0, 0.8)
        
        # 4. Machine speed with possible operator intervention
        speed_target = machine_speed_sp[k]
        if event_type == 'waterlogging' and moisture[k-1] > 6.5:
            # Operator slowdown to dry paper: reduce speed setpoint dynamically
            speed_target -= (100.0 + np.random.normal(0, 15.0)) * (moisture[k-1] - 6.5)
            speed_target = max(speed_target, RECIPE_LIMITS['machine_speed']['min'])
            
        alpha_speed = dt / (tau_speed + dt)
        machine_speed[k] = machine_speed[k-1] + alpha_speed * (speed_target - machine_speed[k-1]) + np.random.normal(0, 0.9)
        
        # 5. Moisture (responds to speed, dryer_temp, stock_flow)
        m_target = (6.0 + 
                    0.015 * (machine_speed[k] - 400.0) - 
                    0.045 * (dryer_temp[k] - 100.0) + 
                    0.002 * (stock_flow[k] - 800.0))
        m_target = np.clip(m_target, 4.0, 9.0)
        moisture[k] = moisture[k-1] + (dt / (20.0 + dt)) * (m_target - moisture[k-1]) + np.random.normal(0, 0.15)
        
        # 6. Ash
        ash_target = 100.0 * (filler_flow[k] * 0.8) / (stock_flow[k] + 1e-5)
        ash[k] = ash[k-1] + (dt / (30.0 + dt)) * (ash_target - ash[k-1]) + np.random.normal(0, 0.1)
        
        # 7. Caliper
        caliper_target = 100.0 + 0.15 * (machine_speed[k] - 400.0) - 8.0 * (steam_pressure[k] - 2.4)
        caliper[k] = caliper[k-1] + (dt / (15.0 + dt)) * (caliper_target - caliper[k-1]) + np.random.normal(0, 0.4)
        
        # 8. Mass flow effective (transport delay)
        lookback = max(0, k - delay_steps)
        target_mass = stock_flow[lookback] * 0.9 + filler_flow[lookback] * 0.1
        alpha_mass = dt / (tau_mass + dt)
        mass_flow_eff[k] = mass_flow_eff[k-1] + alpha_mass * (target_mass - mass_flow_eff[k-1])
        
        # 9. Basis weight (actual) with sensor noise & turbulence
        bw_target = 45.0 * mass_flow_eff[k] / machine_speed[k]
        alpha_bw = dt / (tau_bw + dt)
        basis_weight[k] = basis_weight[k-1] + alpha_bw * (bw_target - basis_weight[k-1]) + np.random.normal(0, 0.12)

    # Event metrics calculation
    deviation = np.abs(basis_weight - basis_weight_sp) / basis_weight_sp
    
    # Monitor deviation starting 60s into the transition to allow normal startup
    monitoring_idx = np.where(t >= 60)[0]
    max_deviation = np.max(deviation[monitoring_idx])
    off_spec = int(max_deviation > 0.025)
    
    # time_to_stabilize: time (s) from t=0 until deviation stays <= 1.5% (0.015) for at least 300s (30 steps)
    time_to_stabilize = 2700.0
    for i in range(np.where(t >= 0)[0][0], len(t)):
        if np.all(deviation[i:] <= 0.015):
            time_to_stabilize = t[i]
            break

    # Build DataFrame for the event
    df_event = pd.DataFrame({
        'event_id': event_id,
        't_seconds': t,
        'grade_from': grade_from,
        'grade_to': grade_to,
        'event_type': event_type,
        'stock_flow_sp': stock_flow_sp,
        'stock_flow': stock_flow,
        'filler_flow_sp': filler_flow_sp,
        'filler_flow': filler_flow,
        'steam_pressure_sp': steam_pressure_sp,
        'steam_pressure': steam_pressure,
        'machine_speed_sp': machine_speed_sp,
        'machine_speed': machine_speed,
        'dryer_temp': dryer_temp,
        'moisture': moisture,
        'ash': ash,
        'caliper': caliper,
        'basis_weight': basis_weight,
        'basis_weight_setpoint': basis_weight_sp,
        'deviation': deviation,
        'recipe_min_stock': RECIPE_LIMITS['stock_flow']['min'],
        'recipe_max_stock': RECIPE_LIMITS['stock_flow']['max'],
        'recipe_min_filler': RECIPE_LIMITS['filler_flow']['min'],
        'recipe_max_filler': RECIPE_LIMITS['filler_flow']['max'],
        'recipe_min_steam': RECIPE_LIMITS['steam_pressure']['min'],
        'recipe_max_steam': RECIPE_LIMITS['steam_pressure']['max'],
        'recipe_min_speed': RECIPE_LIMITS['machine_speed']['min'],
        'recipe_max_speed': RECIPE_LIMITS['machine_speed']['max'],
        'recipe_min_moisture': RECIPE_LIMITS['moisture']['min'],
        'recipe_max_moisture': RECIPE_LIMITS['moisture']['max'],
        'recipe_min_ash': RECIPE_LIMITS['ash']['min'],
        'recipe_max_ash': RECIPE_LIMITS['ash']['max'],
        'recipe_min_caliper': RECIPE_LIMITS['caliper']['min'],
        'recipe_max_caliper': RECIPE_LIMITS['caliper']['max'],
        'recipe_min_bw': RECIPE_LIMITS['basis_weight']['min'],
        'recipe_max_bw': RECIPE_LIMITS['basis_weight']['max'],
        'off_spec_flag': off_spec,
        'stabilize_time_s': time_to_stabilize
    })
    
    return df_event

def generate_dataset(num_events=200):
    np.random.seed(42)
    events_list = []
    
    transitions = [('G1', 'G2'), ('G2', 'G3'), ('G3', 'G2'), ('G2', 'G1'), ('G1', 'G3'), ('G3', 'G1')]
    
    print(f"Generating {num_events} grade-change events...")
    
    for i in range(num_events):
        event_id = f"E{i+1:03d}"
        grade_from, grade_to = transitions[i % len(transitions)]
        
        rand_val = np.random.rand()
        if rand_val < 0.65:
            event_type = 'normal'
        elif rand_val < 0.85:
            event_type = 'waterlogging'
        else:
            event_type = 'speed_mismatch'
            
        seed = 1000 + i
        df_event = generate_event(event_id, grade_from, grade_to, event_type, seed)
        
        # Introduce subtle real-world sensor drift / label noise for 2 events
        if i in [18, 143]:
            df_event['off_spec_flag'] = 1 - df_event['off_spec_flag'].iloc[0]
            
        events_list.append(df_event)
        
    df_all = pd.concat(events_list, ignore_index=True)
    
    # Save to CSV
    csv_path = os.path.join(DATA_DIR, "historian_data.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"Dataset generated successfully and saved to: {csv_path}")
    print(f"Total rows: {len(df_all)}")
    
    # Print summary statistics of events
    df_summary = df_all.groupby('event_id').first().reset_index()
    off_spec_pct = df_summary['off_spec_flag'].mean() * 100
    mean_stab_time = df_summary['stabilize_time_s'].mean()
    
    print(f"\n--- Dataset Summary ---")
    print(f"Off-spec events: {off_spec_pct:.1f}%")
    print(f"Mean stabilization time: {mean_stab_time:.1f} seconds")
    print(f"Event types breakdown:")
    print(df_summary['event_type'].value_counts())
    
    # Plot example transitions
    plot_example_events(df_all)
    
def plot_example_events(df):
    plt.figure(figsize=(15, 10))
    sns.set_theme(style="darkgrid")
    
    # Find a G1->G2 transition for normal and problem events
    g1_to_g2 = df[(df['grade_from'] == 'G1') & (df['grade_to'] == 'G2')]
    
    # Normal event example
    normal_events = g1_to_g2[g1_to_g2['event_type'] == 'normal']['event_id'].unique()
    # Bad event example
    waterlog_events = g1_to_g2[g1_to_g2['event_type'] == 'waterlogging']['event_id'].unique()
    mismatch_events = g1_to_g2[g1_to_g2['event_type'] == 'speed_mismatch']['event_id'].unique()
    
    if len(normal_events) > 0 and len(waterlog_events) > 0 and len(mismatch_events) > 0:
        ev_normal = normal_events[0]
        ev_waterlog = waterlog_events[0]
        ev_mismatch = mismatch_events[0]
        
        df_norm = df[df['event_id'] == ev_normal]
        df_wl = df[df['event_id'] == ev_waterlog]
        df_mm = df[df['event_id'] == ev_mismatch]
        
        # Subplot 1: Basis Weight comparison
        plt.subplot(2, 2, 1)
        plt.plot(df_norm['t_seconds'], df_norm['basis_weight'], 'g-', label=f'Normal ({ev_normal})')
        plt.plot(df_wl['t_seconds'], df_wl['basis_weight'], 'r-', label=f'Waterlogging ({ev_waterlog})')
        plt.plot(df_mm['t_seconds'], df_mm['basis_weight'], 'b-', label=f'Speed Mismatch ({ev_mismatch})')
        plt.plot(df_norm['t_seconds'], df_norm['basis_weight_setpoint'], 'k--', label='Setpoint')
        plt.fill_between(df_norm['t_seconds'], 
                         df_norm['basis_weight_setpoint'] * 0.975, 
                         df_norm['basis_weight_setpoint'] * 1.025, 
                         color='blue', alpha=0.1, label='±2.5% Tolerance Band')
        plt.title('Basis Weight Trajectories (G1 -> G2)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Basis Weight (g/m²)')
        plt.legend()
        
        # Subplot 2: Dryer Temp vs Steam Pressure
        plt.subplot(2, 2, 2)
        plt.plot(df_norm['t_seconds'], df_norm['dryer_temp'], 'g-', label='Dryer Temp (Normal)')
        plt.plot(df_wl['t_seconds'], df_wl['dryer_temp'], 'r-', label='Dryer Temp (Waterlog)')
        plt.plot(df_norm['t_seconds'], df_norm['steam_pressure'] * 50.0, 'k--', label='Expected Temp (50 * Steam)')
        plt.title('Dryer Temperature Response')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Temperature / Press (scaled)')
        plt.legend()
        
        # Subplot 3: Moisture Response
        plt.subplot(2, 2, 3)
        plt.plot(df_norm['t_seconds'], df_norm['moisture'], 'g-', label='Moisture (Normal)')
        plt.plot(df_wl['t_seconds'], df_wl['moisture'], 'r-', label='Moisture (Waterlog)')
        plt.plot(df_norm['t_seconds'], [RECIPE_LIMITS['moisture']['max']]*len(df_norm), 'k:', label='Recipe Max Moisture (8%)')
        plt.title('Moisture Trajectories')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Moisture (%)')
        plt.legend()
        
        # Subplot 4: Machine Speed & Stock Flow
        plt.subplot(2, 2, 4)
        plt.plot(df_norm['t_seconds'], df_norm['machine_speed'], 'g-', label='Speed (Normal)')
        plt.plot(df_wl['t_seconds'], df_wl['machine_speed'], 'r-', label='Speed (Waterlog)')
        plt.plot(df_mm['t_seconds'], df_mm['machine_speed'], 'b-', label='Speed (Mismatch)')
        plt.title('Machine Speed Adjustments')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (m/min)')
        plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(DATA_DIR, "example_events.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Sanity check plot saved to: {plot_path}")

if __name__ == "__main__":
    generate_dataset()
