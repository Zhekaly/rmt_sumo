import traci
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from tensorflow import keras
import joblib
import time

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
CONFIG_FILE = r"C:\sumodemo\net\cross.sumocfg"

# ============================================
# FUNCTIONS FOR ADAPTIVE TRAFFIC LIGHT
# ============================================

def load_models():
    """Loads trained ML models"""
    try:
        cat_model = CatBoostRegressor()
        cat_model.load_model("catboost_model.cbm")
        nn_model = keras.models.load_model("nn_model.keras")
        scaler = joblib.load("scaler.pkl")
        return cat_model, nn_model, scaler
    except Exception as e:
        print(f"⚠️ Failed to load models: {e}")
        print("   Run train_models.py first")
        return None, None, None

def get_junction_lanes(junction_id):
    """Gets incoming lanes for the junction"""
    # Get all edges leading to the junction
    incoming_edges = []
    
    # Using information from net.xml
    junction_map = {
        'j_0_0': {
            'NS': ['e_in_top_0', 'e_v_0_1_to_0_0'],
            'EW': ['e_in_left_0', 'e_h_1_0_to_0_0']
        },
        'j_1_0': {
            'NS': ['e_in_top_1', 'e_v_1_1_to_1_0'],
            'EW': ['e_h_0_0_to_1_0', 'e_h_2_0_to_1_0']
        },
        'j_2_0': {
            'NS': ['e_in_top_2', 'e_v_2_1_to_2_0'],
            'EW': ['e_h_1_0_to_2_0', 'e_in_right_0']
        },
        'j_0_1': {
            'NS': ['e_v_0_0_to_0_1', 'e_v_0_2_to_0_1'],
            'EW': ['e_in_left_1', 'e_h_1_1_to_0_1']
        },
        'j_1_1': {
            'NS': ['e_v_1_0_to_1_1', 'e_v_1_2_to_1_1'],
            'EW': ['e_h_0_1_to_1_1', 'e_h_2_1_to_1_1']
        },
        'j_2_1': {
            'NS': ['e_v_2_0_to_2_1', 'e_v_2_2_to_2_1'],
            'EW': ['e_h_1_1_to_2_1', 'e_in_right_1']
        },
        'j_0_2': {
            'NS': ['e_v_0_1_to_0_2', 'e_in_bot_0'],
            'EW': ['e_in_left_2', 'e_h_1_2_to_0_2']
        },
        'j_1_2': {
            'NS': ['e_v_1_1_to_1_2', 'e_in_bot_1'],
            'EW': ['e_h_0_2_to_1_2', 'e_h_2_2_to_1_2']
        },
        'j_2_2': {
            'NS': ['e_v_2_1_to_2_2', 'e_in_bot_2'],
            'EW': ['e_h_1_2_to_2_2', 'e_in_right_2']
        }
    }
    
    return junction_map.get(junction_id, {'NS': [], 'EW': []})

def get_edge_features(edge_ids):
    """Gets features for a list of edges"""
    edge_data = []
    
    for edge_id in edge_ids:
        try:
            # Get all lanes of this edge
            lanes = traci.edge.getLaneNumber(edge_id)
            
            for lane_idx in range(lanes):
                lane_id = f"{edge_id}_{lane_idx}"
                
                veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                halting = traci.lane.getLastStepHaltingNumber(lane_id)
                
                CO2 = 0
                for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    CO2 += traci.vehicle.getCO2Emission(veh_id)
                
                # Using more informative features
                edge_data.append([veh_count, waiting_time, CO2, halting])
        except:
            pass
    
    if edge_data:
        # Aggregate across all lanes in the direction
        features = np.mean(edge_data, axis=0)
        # Return only first 3 features for model compatibility
        return features[:3].reshape(1, -1)
    
    return np.array([[0, 0, 0]])

def predict_green_time_adaptive(features, cat_model, nn_model, scaler, base_time=30):
    """
    Improved prediction function considering:
    1. Base time
    2. Current load
    3. Change constraints
    """
    X_scaled = scaler.transform(features)
    
    cat_pred = cat_model.predict(features)[0]
    nn_pred = nn_model.predict(X_scaled, verbose=0)[0][0]
    
    # Ensemble prediction
    ml_prediction = (cat_pred + nn_pred) / 2
    
    # Adaptive correction based on load
    veh_count, waiting_time, co2 = features[0]
    
    # If many vehicles - increase time
    if veh_count > 10 or waiting_time > 30:
        ml_prediction *= 1.3
    elif veh_count > 5 or waiting_time > 15:
        ml_prediction *= 1.15
    
    # Combine with base time (70% ML, 30% base)
    optimal_time = 0.7 * ml_prediction + 0.3 * base_time
    
    # Wider range
    return max(20, min(90, optimal_time))

# ============================================
# SIMULATION WITH FIXED TRAFFIC LIGHT
# ============================================

def run_fixed_traffic_light(duration=3600):
    """Runs simulation with fixed traffic light"""
    print("\n" + "="*60)
    print("🚦 SIMULATION #1: FIXED TRAFFIC LIGHT")
    print("="*60)
    print("⏱️  Phases: 42s green NS → 3s yellow → 42s green EW → 3s yellow")
    print(f"⏱️  Duration: {duration} seconds ({duration//60} minutes)")
    print("-"*60)
    
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start", "--quit-on-end", "--no-warnings"])
    
    metrics = {
        'waiting_times': [],
        'queue_lengths': [],
        'co2_emissions': [],
        'served_vehicles': 0,
        'avg_speed': []
    }
    
    start_time = time.time()
    
    for step in range(duration):
        traci.simulationStep()
        
        # Collect metrics every second
        total_waiting = 0
        total_queue = 0
        total_co2 = 0
        total_speed = 0
        vehicle_count = 0
        
        for lane_id in traci.lane.getIDList():
            total_waiting += traci.lane.getWaitingTime(lane_id)
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                total_co2 += traci.vehicle.getCO2Emission(veh_id)
                total_speed += traci.vehicle.getSpeed(veh_id)
                vehicle_count += 1
        
        metrics['waiting_times'].append(total_waiting)
        metrics['queue_lengths'].append(total_queue)
        metrics['co2_emissions'].append(total_co2)
        if vehicle_count > 0:
            metrics['avg_speed'].append(total_speed / vehicle_count)
        
        # Progress every 10%
        if step % (duration // 10) == 0 and step > 0:
            progress = (step / duration) * 100
            print(f"   {progress:5.1f}% | Wait: {total_waiting:7.1f}s | Queue: {total_queue:3d} | CO2: {total_co2:8.1f}g")
    
    # Count served vehicles
    metrics['served_vehicles'] = traci.simulation.getArrivedNumber()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Simulation completed in {elapsed:.1f} seconds")
    
    traci.close()
    return metrics

# ============================================
# SIMULATION WITH ADAPTIVE TRAFFIC LIGHT
# ============================================

def run_adaptive_traffic_light(duration=3600):
    """Runs simulation with adaptive traffic light"""
    print("\n" + "="*60)
    print("🤖 SIMULATION #2: ADAPTIVE TRAFFIC LIGHT (ML)")
    print("="*60)
    print("🧠 Using ML models for dynamic control")
    print(f"⏱️  Duration: {duration} seconds ({duration//60} minutes)")
    print("-"*60)
    
    # Load models
    cat_model, nn_model, scaler = load_models()
    if cat_model is None:
        print("❌ Cannot run adaptive traffic light without models!")
        return None
    
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start", "--quit-on-end", "--no-warnings"])
    
    metrics = {
        'waiting_times': [],
        'queue_lengths': [],
        'co2_emissions': [],
        'served_vehicles': 0,
        'avg_speed': [],
        'phase_changes': [],
        'green_times': []
    }
    
    # Get list of traffic lights
    tls_list = traci.trafficlight.getIDList()
    
    # State for each traffic light
    tls_states = {}
    for tls_id in tls_list:
        tls_states[tls_id] = {
            'current_phase': 0,
            'phase_timer': 0,
            'phase_durations': [35, 3, 35, 3],  # Initial values
            'lanes': get_junction_lanes(tls_id)
        }
    
    start_time = time.time()
    
    for step in range(duration):
        traci.simulationStep()
        
        # Update each traffic light
        for tls_id, state in tls_states.items():
            state['phase_timer'] += 1
            
            # Check if phase needs to be switched
            if state['phase_timer'] >= state['phase_durations'][state['current_phase']]:
                current_phase = state['current_phase']
                
                # Before green phase, predict time
                if current_phase == 1:  # Transition from yellow NS to green EW
                    edge_ids = state['lanes']['EW']
                    features = get_edge_features(edge_ids)
                    green_time = predict_green_time_adaptive(
                        features, cat_model, nn_model, scaler, base_time=35
                    )
                    state['phase_durations'][2] = green_time
                    metrics['green_times'].append(('EW', green_time))
                    
                elif current_phase == 3:  # Transition from yellow EW to green NS
                    edge_ids = state['lanes']['NS']
                    features = get_edge_features(edge_ids)
                    green_time = predict_green_time_adaptive(
                        features, cat_model, nn_model, scaler, base_time=35
                    )
                    state['phase_durations'][0] = green_time
                    metrics['green_times'].append(('NS', green_time))
                
                # Switch phase
                state['current_phase'] = (current_phase + 1) % 4
                try:
                    traci.trafficlight.setPhase(tls_id, state['current_phase'])
                except:
                    pass
                state['phase_timer'] = 0
                metrics['phase_changes'].append((step, tls_id))
        
        # Collect metrics
        total_waiting = 0
        total_queue = 0
        total_co2 = 0
        total_speed = 0
        vehicle_count = 0
        
        for lane_id in traci.lane.getIDList():
            total_waiting += traci.lane.getWaitingTime(lane_id)
            total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
            
            for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                total_co2 += traci.vehicle.getCO2Emission(veh_id)
                total_speed += traci.vehicle.getSpeed(veh_id)
                vehicle_count += 1
        
        metrics['waiting_times'].append(total_waiting)
        metrics['queue_lengths'].append(total_queue)
        metrics['co2_emissions'].append(total_co2)
        if vehicle_count > 0:
            metrics['avg_speed'].append(total_speed / vehicle_count)
        
        # Progress every 10%
        if step % (duration // 10) == 0 and step > 0:
            progress = (step / duration) * 100
            avg_green = np.mean([t for _, t in metrics['green_times'][-20:]]) if metrics['green_times'] else 35
            print(f"   {progress:5.1f}% | Wait: {total_waiting:7.1f}s | Queue: {total_queue:3d} | Avg Green: {avg_green:.1f}s")
    
    metrics['served_vehicles'] = traci.simulation.getArrivedNumber()
    
    elapsed = time.time() - start_time
    print(f"\n✅ Simulation completed in {elapsed:.1f} seconds")
    print(f"   Phase switches: {len(metrics['phase_changes'])}")
    if metrics['green_times']:
        print(f"   Average green duration: {np.mean([t for _, t in metrics['green_times']]):.1f}s")
    
    traci.close()
    return metrics

# ============================================
# COMPARISON AND VISUALIZATION
# ============================================

def compare_metrics(fixed_metrics, adaptive_metrics):
    """Compares metrics of two traffic lights"""
    print("\n" + "="*70)
    print("📊 COMPARISON RESULTS")
    print("="*70)
    
    stats = {
        'Metric': [],
        'Fixed': [],
        'Adaptive': [],
        'Improvement': []
    }
    
    # Average waiting time
    fixed_wait = np.mean(fixed_metrics['waiting_times'])
    adaptive_wait = np.mean(adaptive_metrics['waiting_times'])
    improvement = ((fixed_wait - adaptive_wait) / fixed_wait) * 100
    stats['Metric'].append('Avg. Waiting Time (s)')
    stats['Fixed'].append(f"{fixed_wait:.1f}")
    stats['Adaptive'].append(f"{adaptive_wait:.1f}")
    stats['Improvement'].append(f"{improvement:+.1f}%")
    
    # Maximum waiting time
    fixed_max = np.max(fixed_metrics['waiting_times'])
    adaptive_max = np.max(adaptive_metrics['waiting_times'])
    improvement = ((fixed_max - adaptive_max) / fixed_max) * 100
    stats['Metric'].append('Max. Waiting Time (s)')
    stats['Fixed'].append(f"{fixed_max:.1f}")
    stats['Adaptive'].append(f"{adaptive_max:.1f}")
    stats['Improvement'].append(f"{improvement:+.1f}%")
    
    # Average queue length
    fixed_queue = np.mean(fixed_metrics['queue_lengths'])
    adaptive_queue = np.mean(adaptive_metrics['queue_lengths'])
    improvement = ((fixed_queue - adaptive_queue) / fixed_queue) * 100
    stats['Metric'].append('Avg. Queue (vehicles)')
    stats['Fixed'].append(f"{fixed_queue:.1f}")
    stats['Adaptive'].append(f"{adaptive_queue:.1f}")
    stats['Improvement'].append(f"{improvement:+.1f}%")
    
    # Served vehicles
    if fixed_metrics['served_vehicles'] > 0:
        improvement = ((adaptive_metrics['served_vehicles'] - fixed_metrics['served_vehicles']) / fixed_metrics['served_vehicles']) * 100
        stats['Metric'].append('Vehicles Served')
        stats['Fixed'].append(f"{fixed_metrics['served_vehicles']}")
        stats['Adaptive'].append(f"{adaptive_metrics['served_vehicles']}")
        stats['Improvement'].append(f"{improvement:+.1f}%")
    
    # Average CO2
    fixed_co2 = np.mean(fixed_metrics['co2_emissions'])
    adaptive_co2 = np.mean(adaptive_metrics['co2_emissions'])
    if fixed_co2 > 0:
        improvement = ((fixed_co2 - adaptive_co2) / fixed_co2) * 100
    else:
        improvement = 0
    stats['Metric'].append('Avg. CO2 (g/s)')
    stats['Fixed'].append(f"{fixed_co2:.1f}")
    stats['Adaptive'].append(f"{adaptive_co2:.1f}")
    stats['Improvement'].append(f"{improvement:+.1f}%")
    
    # Average speed
    if fixed_metrics['avg_speed'] and len(fixed_metrics['avg_speed']) > 0:
        fixed_speed = np.mean(fixed_metrics['avg_speed']) * 3.6
        adaptive_speed = np.mean(adaptive_metrics['avg_speed']) * 3.6
        improvement = ((adaptive_speed - fixed_speed) / fixed_speed) * 100
        stats['Metric'].append('Avg. Speed (km/h)')
        stats['Fixed'].append(f"{fixed_speed:.1f}")
        stats['Adaptive'].append(f"{adaptive_speed:.1f}")
        stats['Improvement'].append(f"{improvement:+.1f}%")
    
    # Display table
    df = pd.DataFrame(stats)
    print("\n" + df.to_string(index=False))
    print("\n" + "="*70)
    
    # Determine winner
    improvements = []
    if fixed_wait > 0:
        improvements.append(((fixed_wait - adaptive_wait) / fixed_wait) * 100)
    if fixed_co2 > 0:
        improvements.append(((fixed_co2 - adaptive_co2) / fixed_co2) * 100)
    if fixed_metrics['avg_speed'] and len(fixed_metrics['avg_speed']) > 0:
        fixed_speed = np.mean(fixed_metrics['avg_speed']) * 3.6
        adaptive_speed = np.mean(adaptive_metrics['avg_speed']) * 3.6
        if fixed_speed > 0:
            improvements.append(((adaptive_speed - fixed_speed) / fixed_speed) * 100)
    
    if len(improvements) > 0:
        avg_improvement = np.mean(improvements)
    else:
        avg_improvement = 0
    
    if avg_improvement > 5:
        print("🏆 WINNER: ADAPTIVE TRAFFIC LIGHT!")
        print(f"   Average improvement: {avg_improvement:.1f}%")
    elif avg_improvement < -5:
        print("🏆 WINNER: FIXED TRAFFIC LIGHT")
        print(f"   It performs better by {-avg_improvement:.1f}%")
    else:
        print("🤝 RESULTS ARE COMPARABLE")
        print(f"   Difference: {avg_improvement:.1f}%")
    
    print("="*70)
    
    # Save results
    df.to_csv('comparison_results.csv', index=False)
    print("\n💾 Results saved to: comparison_results.csv")
    
    return stats

def plot_comparison(fixed_metrics, adaptive_metrics):
    """Creates comparison charts"""
    print("\n📊 Creating charts...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Fixed vs Adaptive Traffic Light Comparison', fontsize=16, fontweight='bold')
    
    time_axis = np.arange(len(fixed_metrics['waiting_times']))
    
    # Chart 1: Waiting time
    ax = axes[0, 0]
    ax.plot(time_axis, fixed_metrics['waiting_times'], label='Fixed', alpha=0.7, linewidth=1)
    ax.plot(time_axis, adaptive_metrics['waiting_times'], label='Adaptive', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Waiting Time (s)')
    ax.set_title('Vehicle Waiting Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chart 2: Queue length
    ax = axes[0, 1]
    ax.plot(time_axis, fixed_metrics['queue_lengths'], label='Fixed', alpha=0.7, linewidth=1)
    ax.plot(time_axis, adaptive_metrics['queue_lengths'], label='Adaptive', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vehicles in Queue')
    ax.set_title('Queue Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chart 3: CO2 emissions
    ax = axes[0, 2]
    ax.plot(time_axis, fixed_metrics['co2_emissions'], label='Fixed', alpha=0.7, linewidth=1)
    ax.plot(time_axis, adaptive_metrics['co2_emissions'], label='Adaptive', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CO2 (g/s)')
    ax.set_title('CO2 Emissions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chart 4: Average speed
    ax = axes[1, 0]
    if fixed_metrics['avg_speed'] and adaptive_metrics['avg_speed']:
        ax.plot(time_axis[:len(fixed_metrics['avg_speed'])], 
                np.array(fixed_metrics['avg_speed']) * 3.6, 
                label='Fixed', alpha=0.7, linewidth=1)
        ax.plot(time_axis[:len(adaptive_metrics['avg_speed'])], 
                np.array(adaptive_metrics['avg_speed']) * 3.6, 
                label='Adaptive', alpha=0.7, linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Average Speed')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chart 5: Waiting time histogram
    ax = axes[1, 1]
    ax.hist(fixed_metrics['waiting_times'], bins=50, alpha=0.6, label='Fixed', color='blue')
    ax.hist(adaptive_metrics['waiting_times'], bins=50, alpha=0.6, label='Adaptive', color='green')
    ax.set_xlabel('Waiting Time (s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Waiting Time Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Chart 6: Metrics summary
    ax = axes[1, 2]
    metrics_names = ['Avg. Wait', 'Avg. Queue', 'Avg. CO2/100', 'Served/10']
    fixed_vals = [
        np.mean(fixed_metrics['waiting_times']),
        np.mean(fixed_metrics['queue_lengths']),
        np.mean(fixed_metrics['co2_emissions']) / 100,
        fixed_metrics['served_vehicles'] / 10
    ]
    adaptive_vals = [
        np.mean(adaptive_metrics['waiting_times']),
        np.mean(adaptive_metrics['queue_lengths']),
        np.mean(adaptive_metrics['co2_emissions']) / 100,
        adaptive_metrics['served_vehicles'] / 10
    ]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    ax.bar(x - width/2, fixed_vals, width, label='Fixed', color='blue', alpha=0.7)
    ax.bar(x + width/2, adaptive_vals, width, label='Adaptive', color='green', alpha=0.7)
    ax.set_ylabel('Value')
    ax.set_title('Metrics Summary')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('traffic_light_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Chart saved: traffic_light_comparison.png")
    plt.close()     

# ============================================
# MAIN FUNCTION
# ============================================

def main():
    print("\n" + "🚦"*30)
    print("FIXED VS ADAPTIVE TRAFFIC LIGHT COMPARISON")
    print("🚦"*30)
    
    # Check configuration
    print("\n⚙️ Checking configuration...")
    print(f"   SUMO: {SUMO_BINARY}")
    print(f"   Config: {CONFIG_FILE}")
    
    import os
    if not os.path.exists(SUMO_BINARY):
        print(f"\n❌ SUMO not found: {SUMO_BINARY}")
        print("   Change SUMO_BINARY path at the beginning of the script")
        return
    
    if not os.path.exists(CONFIG_FILE):
        print(f"\n❌ Configuration not found: {CONFIG_FILE}")
        return
    
    duration = 1800  # 30 minutes (was 3600 - too long for initial test)
    print(f"\n⏱️ Simulation duration: {duration} seconds ({duration//60} minutes)")
    print("   💡 Tip: Increase to 3600 for more accurate results")
    
    # Run both simulations
    fixed_metrics = run_fixed_traffic_light(duration)
    adaptive_metrics = run_adaptive_traffic_light(duration)
    
    if adaptive_metrics is None:
        print("\n❌ Failed to complete comparison")
        return
    
    # Check if there is data
    if len(fixed_metrics['waiting_times']) == 0:
        print("\n❌ Failed to collect metrics!")
        print("   Possible reasons:")
        print("   1. Issues with cross.rou.xml file (no vehicle flows)")
        print("   2. Simulation too short")
        return
    
    # Compare results
    compare_metrics(fixed_metrics, adaptive_metrics)
    
    # Build charts
    plot_comparison(fixed_metrics, adaptive_metrics)
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE!")
    print("="*70)
    print("\n📁 Created files:")
    print("   • comparison_results.csv - metrics table")
    print("   • traffic_light_comparison.png - comparison charts")
    print("\n💡 Recommendation: Analyze the charts to draw conclusions about effectiveness")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Simulation interrupted by user")
        try:
            traci.close()
        except:
            pass
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            traci.close()
        except:
            pass