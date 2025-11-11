import traci
import joblib
import numpy as np
from catboost import CatBoostRegressor
from tensorflow import keras
import time
from collections import defaultdict

SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe"
CONFIG_FILE = r"C:\sumodemo\net\cross.sumocfg"

print("🚀 Loading models...")
try:
    cat_model = CatBoostRegressor()
    cat_model.load_model("catboost_model.cbm")
    nn_model = keras.models.load_model("nn_model.keras")
    scaler = joblib.load("scaler.pkl")
    print("✅ Models loaded!")
except Exception as e:
    print(f"❌ Model loading error: {e}")
    print("   First train models using train_models.py")
    exit(1)

class IntersectionController:
    """Controller for one intersection"""
    
    def __init__(self, tls_id, incoming_lanes):
        self.tls_id = tls_id
        self.incoming_lanes = incoming_lanes  # {direction: [lane_ids]}
        self.current_phase = 0
        self.phase_timer = 0
        self.phase_durations = [42, 3, 42, 3]  # NS green, yellow, EW green, yellow
        self.stats = {
            'total_waiting': 0,
            'phase_switches': 0,
            'vehicles_served': 0
        }
        self.tls_exists = False
        self.valid_lanes = []
        
    def initialize(self):
        """Initialize and validate traffic light"""
        try:
            # Check if traffic light exists
            all_tls = traci.trafficlight.getIDList()
            if self.tls_id not in all_tls:
                print(f"⚠️  Traffic light {self.tls_id} not found in simulation")
                return False
            
            self.tls_exists = True
            
            # Validate lanes
            all_lanes = set(traci.lane.getIDList())
            for direction, lanes in self.incoming_lanes.items():
                valid = [l for l in lanes if l in all_lanes]
                if valid:
                    self.incoming_lanes[direction] = valid
                else:
                    print(f"⚠️  No valid lanes for {self.tls_id} direction {direction}")
            
            # Get valid phase count
            try:
                program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                num_phases = len(program.phases)
                print(f"✅ {self.tls_id}: {num_phases} phases detected")
            except:
                print(f"⚠️  Could not get phase info for {self.tls_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing {self.tls_id}: {e}")
            return False
    
    def get_direction_features(self, direction):
        """Gets aggregated features for direction"""
        lanes = self.incoming_lanes.get(direction, [])
        if not lanes:
            return np.array([[0, 0, 0]])
        
        lane_data = []
        for lane_id in lanes:
            try:
                veh_count = traci.lane.getLastStepVehicleNumber(lane_id)
                waiting_time = traci.lane.getWaitingTime(lane_id)
                
                CO2 = 0
                for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    CO2 += traci.vehicle.getCO2Emission(veh_id)
                
                lane_data.append([veh_count, waiting_time, CO2])
            except Exception as e:
                pass
        
        if lane_data:
            return np.mean(lane_data, axis=0).reshape(1, -1)
        return np.array([[0, 0, 0]])
    
    def predict_green_time(self, features):
        """Predicts optimal green time"""
        try:
            X_scaled = scaler.transform(features)
            
            cat_pred = cat_model.predict(features)[0]
            nn_pred = nn_model.predict(X_scaled, verbose=0)[0][0]
            
            optimal_time = (cat_pred + nn_pred) / 2
            return max(10, min(60, optimal_time))
        except Exception as e:
            print(f"⚠️  Prediction error: {e}")
            return 30  # Default fallback
    
    def update(self, step):
        """Updates traffic light state"""
        if not self.tls_exists:
            return False
            
        self.phase_timer += 1
        
        # Check if phase needs switching
        if self.phase_timer >= self.phase_durations[self.current_phase]:
            # Predict time before green phase
            if self.current_phase == 1:  # Transition from yellow NS to green EW
                features = self.get_direction_features('EW')
                self.phase_durations[2] = self.predict_green_time(features)
                
            elif self.current_phase == 3:  # Transition from yellow EW to green NS
                features = self.get_direction_features('NS')
                self.phase_durations[0] = self.predict_green_time(features)
            
            # Switch phase
            try:
                self.current_phase = (self.current_phase + 1) % 4
                traci.trafficlight.setPhase(self.tls_id, self.current_phase)
                self.phase_timer = 0
                self.stats['phase_switches'] += 1
                return True  # Phase changed
            except Exception as e:
                print(f"⚠️  Error switching phase for {self.tls_id}: {e}")
                return False
        
        return False
    
    def get_current_stats(self):
        """Returns current intersection statistics"""
        waiting_time = 0
        queue_length = 0
        
        for lanes in self.incoming_lanes.values():
            for lane_id in lanes:
                try:
                    waiting_time += traci.lane.getWaitingTime(lane_id)
                    queue_length += traci.lane.getLastStepHaltingNumber(lane_id)
                except:
                    pass
        
        self.stats['total_waiting'] = waiting_time
        return waiting_time, queue_length


class NetworkController:
    """Controller for entire intersection network"""
    
    def __init__(self):
        self.intersections = {}
        self.global_stats = {
            'total_waiting': [],
            'total_queues': [],
            'total_co2': [],
            'vehicles_in_network': []
        }
    
    def setup_intersections(self):
        """Sets up controllers for all intersections"""
        
        print("\n🔍 Detecting available lanes...")
        all_lanes = traci.lane.getIDList()
        print(f"   Found {len(all_lanes)} lanes in simulation")
        
        # Define incoming lanes for each intersection
        # Using simplified edge-based approach (without lane indices)
        intersection_configs = {
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
        
        # Expand edge IDs to include all lanes
        expanded_configs = {}
        for tls_id, directions in intersection_configs.items():
            expanded_configs[tls_id] = {'NS': [], 'EW': []}
            
            for direction, edge_ids in directions.items():
                for edge_id in edge_ids:
                    # Find all lanes for this edge
                    matching_lanes = [l for l in all_lanes if l.startswith(edge_id + '_')]
                    expanded_configs[tls_id][direction].extend(matching_lanes)
        
        # Create controllers for each intersection
        print("\n🚦 Setting up intersection controllers...")
        active_count = 0
        for tls_id, lanes_config in expanded_configs.items():
            controller = IntersectionController(tls_id, lanes_config)
            if controller.initialize():
                self.intersections[tls_id] = controller
                active_count += 1
        
        print(f"✅ {active_count}/{len(expanded_configs)} intersections active")
        
        if active_count == 0:
            print("❌ No active intersections! Check your network configuration.")
            return False
        
        return True
    
    def update_all(self, step):
        """Updates all intersections"""
        phase_changes = []
        
        for tls_id, controller in self.intersections.items():
            try:
                changed = controller.update(step)
                if changed:
                    phase_changes.append(tls_id)
            except Exception as e:
                pass  # Silently continue if one intersection fails
        
        return phase_changes
    
    def collect_global_stats(self):
        """Collects global network statistics"""
        total_waiting = 0
        total_queue = 0
        total_co2 = 0
        vehicle_count = 0
        
        # Collect data from all lanes
        for lane_id in traci.lane.getIDList():
            try:
                total_waiting += traci.lane.getWaitingTime(lane_id)
                total_queue += traci.lane.getLastStepHaltingNumber(lane_id)
                
                for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    total_co2 += traci.vehicle.getCO2Emission(veh_id)
                    vehicle_count += 1
            except:
                pass
        
        self.global_stats['total_waiting'].append(total_waiting)
        self.global_stats['total_queues'].append(total_queue)
        self.global_stats['total_co2'].append(total_co2)
        self.global_stats['vehicles_in_network'].append(vehicle_count)
        
        return total_waiting, total_queue, total_co2, vehicle_count
    
    def print_summary(self, step, total_waiting, total_queue, total_co2, vehicle_count):
        """Prints network summary"""
        print(f"\n{'='*80}")
        print(f"📊 STEP {step:5d} | NETWORK STATISTICS")
        print(f"{'='*80}")
        print(f"  🚗 Vehicles in network: {vehicle_count:4d}")
        print(f"  ⏳ Total waiting: {total_waiting:8.1f} sec")
        print(f"  🚦 Vehicles in queue: {total_queue:4d}")
        print(f"  💨 CO2: {total_co2:10.1f} g")
        
        # Intersection statistics
        print(f"\n  🔍 INTERSECTIONS:")
        for tls_id, controller in sorted(self.intersections.items()):
            waiting, queue = controller.get_current_stats()
            phase_name = ['NS🟢', 'NS🟡', 'EW🟢', 'EW🟡'][controller.current_phase]
            print(f"    {tls_id}: {phase_name} | Wait: {waiting:6.1f}s | Queue: {queue:2d}")
        
        print(f"{'='*80}")


def run_simulation(duration=3600):
    """Runs simulation with adaptive control"""
    print("\n" + "🚦"*40)
    print("ADAPTIVE URBAN NETWORK CONTROL (9 INTERSECTIONS)")
    print("🚦"*40)
    
    print(f"\n⚙️  Configuration:")
    print(f"   • Intersections: 9 (3×3 grid)")
    print(f"   • Road types: highways (3 lanes), secondary (2 lanes), local (1 lane)")
    print(f"   • Vehicle types: cars, buses, trucks, emergency")
    print(f"   • Duration: {duration} sec ({duration//60} min)")
    
    print("\n🚀 Starting SUMO...")
    traci.start([SUMO_BINARY, "-c", CONFIG_FILE, "--start", "--quit-on-end", "--no-warnings"])
    
    # Initialize network controller
    network = NetworkController()
    
    if not network.setup_intersections():
        print("❌ Failed to initialize network")
        traci.close()
        return None
    
    print("✅ Network initialized!")
    print("\n🔄 Starting adaptive control...\n")
    
    start_time = time.time()
    report_interval = 300  # Report every 5 minutes
    
    for step in range(duration):
        try:
            traci.simulationStep()
            
            # Update all intersections
            phase_changes = network.update_all(step)
            
            # Collect global statistics
            total_waiting, total_queue, total_co2, vehicle_count = network.collect_global_stats()
            
            # Periodic report
            if step > 0 and step % report_interval == 0:
                network.print_summary(step, total_waiting, total_queue, total_co2, vehicle_count)
            
            # Brief progress every 10%
            elif step % (duration // 10) == 0 and step > 0:
                progress = (step / duration) * 100
                print(f"  {progress:5.1f}% | Vehicles: {vehicle_count:4d} | Wait: {total_waiting:8.1f}s | CO2: {total_co2:10.1f}g")
        
        except traci.exceptions.FatalTraCIError as e:
            print(f"\n❌ TraCI Error at step {step}: {e}")
            print("   Simulation may have ended early or there's a configuration issue")
            break
        except Exception as e:
            print(f"\n⚠️  Error at step {step}: {e}")
            continue
    
    # Final statistics
    elapsed = time.time() - start_time
    served_vehicles = traci.simulation.getArrivedNumber()
    
    print("\n" + "="*80)
    print("🏁 SIMULATION COMPLETE")
    print("="*80)
    print(f"  ⏱️  Execution time: {elapsed:.1f} sec")
    print(f"  🚗 Vehicles served: {served_vehicles}")
    
    if network.global_stats['total_waiting']:
        print(f"  📈 Average waiting: {np.mean(network.global_stats['total_waiting']):.1f} sec")
        print(f"  📉 Average queue: {np.mean(network.global_stats['total_queues']):.1f} vehicles")
        print(f"  💨 Average CO2: {np.mean(network.global_stats['total_co2']):.1f} g/sec")
    
    total_switches = sum(c.stats['phase_switches'] for c in network.intersections.values())
    print(f"  🔄 Phase switches: {total_switches}")
    print("="*80)
    
    traci.close()
    
    return network.global_stats


if __name__ == "__main__":
    try:
        stats = run_simulation(duration=3600)
        
        if stats:
            print("\n💾 Statistics collected!")
            print("   Use data for further analysis")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Simulation interrupted by user")
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