#  Adaptive Traffic Light Control System

**Intelligent urban intersection management system based on machine learning**

---

##  Project Overview

This project demonstrates the application of machine learning for optimizing traffic light operations in urban road networks. The system uses the SUMO (Simulation of Urban MObility) simulator to model traffic and adaptively controls green phase durations based on real-time road conditions.

###  Key Features

- **Hybrid ML Approach**: Combination of CatBoost and Neural Networks for optimal timing predictions
- **Real-Time Control**: Dynamic adaptation of traffic lights to current road load
- **Complex Topology**: 3×3 intersection network with various road types (highways, secondary, local)
- **Multiple Vehicle Types**: Cars, buses, trucks, emergency vehicles
- **Comparative Analysis**: Built-in tools for evaluating effectiveness vs fixed-time traffic lights

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SUMO Simulator                        │
│  (Urban traffic network 3×3, 9 intersections)           │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│              Data Collection Layer                       │
│  • Vehicle counts      • CO2 emissions                  │
│  • Waiting times       • Queue lengths                  │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│           Machine Learning Models                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │  CatBoost    │  │  Neural Net  │  │   Ensemble   │ │
│  │  Regressor   │  │  (Keras)     │  │   Model      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│         Adaptive Traffic Light Controller               │
│  • Real-time decisions  • Phase management              │
│  • Multi-intersection coordination                      │
└─────────────────────────────────────────────────────────┘
```

---

##  Installation

### Requirements

- **Python 3.8+**
- **SUMO 1.20+** ([download](https://www.eclipse.org/sumo/))
- Python libraries (see below)

### 1. Installing SUMO

**Windows:**
```bash
# Download the installer from the official website
# https://www.eclipse.org/sumo/
# After installation, add the path to PATH or specify it in scripts
```

**Linux (Ubuntu/Debian):**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

**macOS:**
```bash
brew install sumo
```

### 2. Installing Python Dependencies

```bash
pip install traci pandas numpy matplotlib scikit-learn catboost tensorflow joblib
```

### 3. Cloning the Project

```bash
git clone https://github.com/Zhekaly/rmt_sumo
cd sumodemo
```

---

##  Quick Start

### Step 1: Collect Traffic Data

```bash
python data_collector.py
```

**What happens:**
- SUMO starts with `cross.sumocfg` configuration
- Data is collected for 3600 seconds (1 hour of simulation)
- `traffic_data.csv` file is saved with metrics

**Output data:** `traffic_data.csv` (~100-200k records)

---

### Step 2: Train Models

```bash
python train_models.py
```

**What happens:**
- Load data from `traffic_data.csv`
- Train CatBoost and Neural Network
- Save trained models

**Output files:**
- `catboost_model.cbm` - CatBoost model
- `nn_model.keras` - Neural Network
- `scaler.pkl` - Feature scaler

---

### Step 3: Evaluate Models

```bash
python evaluate_models.py
```

**What happens:**
- Cross-validation of models
- Calculate metrics (MAE, RMSE, R²)
- Visualize results

**Output files:**
- `model_evaluation_fixed.png` - performance charts
- `model_predictions_fixed.csv` - detailed predictions

---

### Step 4: Run Adaptive Controller

```bash
python online_controller.py
```

**What happens:**
- Start SUMO with graphical interface
- Control 9 intersections in real-time
- Adaptive phase duration adjustment

**Expected results:**
- 15-25% reduction in waiting time
- 10-15% decrease in CO₂ emissions
- 8-12% increase in average speed

---

### Step 5: Compare with Fixed-Time Signals

```bash
python compare_traffic_lights.py
```

**What happens:**
- Run two simulations: fixed vs adaptive
- Collect metrics for both systems
- Generate comparative charts

**Output files:**
- `comparison_results.csv` - metrics table
- `traffic_light_comparison.png` - visual comparison

---

##  Data Structure

### Input Features

| Feature | Description | Units |
|---------|-------------|-------|
| `veh_count` | Number of vehicles on lane | count |
| `waiting_time` | Total waiting time | sec |
| `CO2` | CO₂ emissions | g/sec |
| `veh_waiting_ratio` | Ratio of count to waiting time | - |
| `CO2_per_vehicle` | Emissions per vehicle | g |
| `traffic_density` | Traffic density | units |

### Target Variable

- **Optimal green phase duration** (5-90 seconds)
- Calculated based on heuristics considering:
  - Direction load
  - Environmental factors
  - Safety

---

##  Configuration

### Setting Paths in Scripts

Before running, specify correct paths:

```python
# In each .py file, modify:
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"  # or sumo-gui.exe
CONFIG_FILE = r"C:\sumodemo\net\cross.sumocfg"
```

### Simulation Parameters

**In `cross.sumocfg`:**
```xml
<time>
    <begin value="0"/>
    <end value="3600"/>      <!-- Duration: 1 hour -->
    <step-length value="1"/>  <!-- Step: 1 sec -->
</time>
```

**In `cross.rou.xml`:**
- Flow intensity (`probability` parameter)
- Vehicle types
- Routes

---

##  Project Structure

```
sumodemo/
│
├── 📜 compare_traffic_lights.py   # Compare fixed vs adaptive systems
├── 📜 data_collector.py           # Collect data from SUMO
├── 📜 evaluate_models.py          # Evaluate ML models
├── 📜 online_controller.py        # Real-time adaptive control
├── 📜 train_models.py             # Train CatBoost + NN
│
├── 📂 net/                        # SUMO network configuration
│   ├── cross.edg.xml              # Road definitions (edges)
│   ├── cross.nod.xml              # Intersections (nodes)
│   ├── cross.rou.xml              # Vehicle routes
│   ├── cross.net.xml              # Compiled network
│   ├── cross.sumocfg              # Main SUMO config
│   ├── cross.netccfg              # netconvert config
│   └── gui-settings.xml           # GUI settings
│
├── 📊 traffic_data.csv            # Collected data (generated)
├── 🤖 catboost_model.cbm          # Trained CatBoost model
├── 🤖 nn_model.keras               # Trained Neural Network
├── 🔧 scaler.pkl                   # Feature scaler
│
├── 📈 model_evaluation_fixed.png  # Model evaluation charts
├── 📈 traffic_light_comparison.png # Comparison charts
├── 📊 comparison_results.csv      # Comparison table
├── 📊 model_predictions_fixed.csv # Model predictions
│
├── 📄 README.md                    # This file
└── 📄 requirements.txt             # Python dependencies
```

---

##  Visualizations

### 1. Model Evaluation (`model_evaluation_fixed.png`)

**6 charts:**
- Predictions vs actual values (CatBoost, NN, Ensemble)
- Error distribution
- Feature importance
- Metrics comparison

### 2. System Comparison (`traffic_light_comparison.png`)

**6 charts:**
- Waiting time (time series)
- Queue length (time series)
- CO₂ emissions (time series)
- Average speed
- Waiting time histogram
- Metrics summary bar chart

---

##  Algorithm Workflow

### 1. Data Collection

```python
for step in simulation:
    for lane in network:
        collect:
            - vehicle_count
            - waiting_time
            - CO2_emissions
            - queue_length
```

### 2. Model Training

```python
# Feature engineering
features = [veh_count, CO2, waiting_ratio, CO2_per_veh, density]
target = optimal_green_time(heuristic)

# Training
catboost_model.fit(features, target)
neural_network.fit(scaled_features, target)
ensemble = (catboost + neural_net) / 2
```

### 3. Adaptive Control

```python
for each intersection:
    # Before phase change
    if yellow_phase_ending:
        features = get_traffic_features(next_direction)
        optimal_time = ensemble_model.predict(features)
        
        # Adaptation with constraints
        green_time = clip(optimal_time, min=20s, max=90s)
        
        # Smooth adaptation
        green_time = 0.7 * ml_prediction + 0.3 * base_time
        
        set_next_phase_duration(green_time)
```

---

##  Troubleshooting

### Issue: "SUMO binary not found"

**Solution:**
```python
# Specify correct path to SUMO
SUMO_BINARY = r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
```

### Issue: "cross.net.xml not found"

**Solution:**
```bash
cd net/
netconvert -c cross.netccfg
```

### Issue: "No vehicles in simulation"

**Causes:**
1. Incorrect routes in `cross.rou.xml`
2. Too low intensity (`probability`)
3. Road ID mismatch

**Solution:**
- Check that `probability` in `cross.rou.xml` > 0.01
- Verify road IDs match in `.rou.xml` and `.net.xml`

### Issue: "Model loading error"

**Solution:**
```bash
# Retrain models
python train_models.py
```

---

##  Theoretical Background

### Problem Statement

**Given:**
- Urban road network with N intersections
- Current traffic state: $S_t = \{v_t, w_t, c_t\}$
  - $v_t$ - vehicle count
  - $w_t$ - waiting time
  - $c_t$ - CO₂ emissions

**Find:**
- Optimal green phase duration $G^*$ to minimize:

$$J = \alpha \cdot \overline{W} + \beta \cdot \overline{Q} + \gamma \cdot \overline{C}$$

Where:
- $\overline{W}$ - average waiting time
- $\overline{Q}$ - average queue length
- $\overline{C}$ - average CO₂ emissions
- $\alpha, \beta, \gamma$ - weight coefficients

### Machine Learning Approach

1. **Regression:** $G^* = f(S_t; \theta)$
2. **Ensemble:**
   $$G^*_{ensemble} = \frac{1}{2}(G^*_{CatBoost} + G^*_{NN})$$

3. **Adaptation:**
   $$G^*_{final} = 0.7 \cdot G^*_{ensemble} + 0.3 \cdot G_{base}$$

---

##  License

This project is licensed under the MIT License. See `LICENSE` file for details.

---

##  Useful Links

- [SUMO Documentation](https://sumo.dlr.de/docs/)
- [TraCI API Reference](https://sumo.dlr.de/docs/TraCI.html)
- [CatBoost Documentation](https://catboost.ai/docs/)
- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)

---

## 🙏 Acknowledgments

- SUMO Development Team for excellent simulator
- ML Community for open-source libraries
- All project contributors

