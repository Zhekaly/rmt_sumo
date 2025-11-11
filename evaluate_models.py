import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from tensorflow import keras
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("📂 Loading data...")
df = pd.read_csv("traffic_data.csv")

print(f"✓ Loaded {len(df)} records")
print(f"✓ Columns: {df.columns.tolist()}")
print(f"\n📊 Data statistics:")
print(df.describe())

if df.isnull().sum().sum() > 0:
    print("\n⚠️  Missing values detected:")
    print(df.isnull().sum())
    df = df.dropna()
    print(f"✓ After removal: {len(df)} records")


print("\n🔧 Feature engineering...")

df_agg = df.groupby('step').agg({
    'veh_count': 'mean',
    'waiting_time': 'mean',
    'CO2': 'mean'
}).reset_index()


df_agg['veh_waiting_ratio'] = df_agg['veh_count'] / (df_agg['waiting_time'] + 1)
df_agg['CO2_per_vehicle'] = df_agg['CO2'] / (df_agg['veh_count'] + 1)
df_agg['traffic_density'] = df_agg['veh_count'] * df_agg['waiting_time']

X = df_agg[["veh_count", "CO2", "veh_waiting_ratio", "CO2_per_vehicle", "traffic_density"]]

def calculate_optimal_green_time(row):
    """
    Heuristic for optimal green light duration
    In reality, these values should be obtained from experiments
    """
    veh = row['veh_count']
    co2 = row['CO2']
    
    base_time = 20
    
    if veh < 3:
        traffic_factor = 0.5
    elif veh < 8:
        traffic_factor = 1.0
    elif veh < 15:
        traffic_factor = 1.5
    else:
        traffic_factor = 2.0
    
    if co2 > 1500:
        eco_factor = 1.2 
    else:
        eco_factor = 1.0
    
    optimal_time = base_time * traffic_factor * eco_factor
    return np.clip(optimal_time, 5, 90)

y = df_agg.apply(calculate_optimal_green_time, axis=1)

print(f"Features: {X.columns.tolist()}")
print(f"Target variable: optimal green time (5-90 sec)")
print(f"Mean: {y.mean():.1f}s, Median: {y.median():.1f}s")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n📊 Data split:")
print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print("\n🤖 Training models...")

print("   → CatBoost (with early stopping)...")
X_train_cb, X_val_cb, y_train_cb, y_val_cb = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

cat_model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    loss_function='MAE',
    early_stopping_rounds=50,
    verbose=False,
    random_state=42
)
cat_model.fit(
    X_train_cb, y_train_cb,
    eval_set=(X_val_cb, y_val_cb),
    verbose=False
)

print("   → Neural Network (with early stopping)...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])

nn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mae',
    metrics=['mae', 'mse']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0
)

print("✅ Training completed!")
print(f"   CatBoost: {cat_model.tree_count_} trees")
print(f"   NN: stopped at epoch {len(history.history['loss'])}")

print("\n🔮 Generating predictions...")

y_pred_cat = cat_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_ensemble = (y_pred_cat + y_pred_nn) / 2

y_pred_cat = np.clip(y_pred_cat, 5, 90)
y_pred_nn = np.clip(y_pred_nn, 5, 90)
y_pred_ensemble = np.clip(y_pred_ensemble, 5, 90)

print("\n" + "="*70)
print("📈 PERFORMANCE METRICS")
print("="*70)

models = {
    'CatBoost': y_pred_cat,
    'Neural Network': y_pred_nn,
    'Ensemble': y_pred_ensemble
}

best_mae = float('inf')
best_model = None

for model_name, y_pred in models.items():
    print(f"\n🤖 {model_name}:")
    print("-" * 70)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    within_5s = np.mean(np.abs(y_test - y_pred) < 5) * 100
    within_10s = np.mean(np.abs(y_test - y_pred) < 10) * 100
    
    print(f"   MAE:           {mae:.2f}s  (mean absolute error)")
    print(f"   RMSE:          {rmse:.2f}s  (penalty for large errors)")
    print(f"   R²:            {r2:.4f}  (explained variance: {r2*100:.1f}%)")
    print(f"   MAPE:          {mape:.2f}%")
    print(f"   Accuracy ±5s:  {within_5s:.1f}%")
    print(f"   Accuracy ±10s: {within_10s:.1f}%")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model_name


print("\n" + "="*70)
print("🔍 FEATURE IMPORTANCE (CatBoost)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feature_importance.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:6.2f}")

print("\n📊 Creating plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Traffic Light Model Performance Evaluation', fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.scatter(y_test, y_pred_cat, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Time (sec)', fontsize=10)
ax.set_ylabel('Predicted Time (sec)', fontsize=10)
ax.set_title(f'CatBoost (MAE={mean_absolute_error(y_test, y_pred_cat):.2f}s)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(y_test, y_pred_nn, alpha=0.6, s=30, color='green', edgecolors='k', linewidths=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Time (sec)', fontsize=10)
ax.set_ylabel('Predicted Time (sec)', fontsize=10)
ax.set_title(f'Neural Network (MAE={mean_absolute_error(y_test, y_pred_nn):.2f}s)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(y_test, y_pred_ensemble, alpha=0.6, s=30, color='purple', edgecolors='k', linewidths=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual Time (sec)', fontsize=10)
ax.set_ylabel('Predicted Time (sec)', fontsize=10)
ax.set_title(f'Ensemble (MAE={mean_absolute_error(y_test, y_pred_ensemble):.2f}s)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
errors_cat = y_test.values - y_pred_cat
errors_nn = y_test.values - y_pred_nn
ax.hist(errors_cat, bins=30, alpha=0.6, label='CatBoost', color='blue')
ax.hist(errors_nn, bins=30, alpha=0.6, label='Neural Net', color='green')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Error (sec)', fontsize=10)
ax.set_ylabel('Frequency', fontsize=10)
ax.set_title('Error Distribution', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
feature_importance.plot(kind='barh', x='feature', y='importance', ax=ax, legend=False, color='teal')
ax.set_xlabel('Importance', fontsize=10)
ax.set_ylabel('')
ax.set_title('Feature Importance (CatBoost)', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

ax = axes[1, 2]
mae_values = [mean_absolute_error(y_test, p) for p in [y_pred_cat, y_pred_nn, y_pred_ensemble]]
r2_values = [r2_score(y_test, p) * 10 for p in [y_pred_cat, y_pred_nn, y_pred_ensemble]]

x = np.arange(3)
width = 0.35

bars1 = ax.bar(x - width/2, mae_values, width, label='MAE (sec)', color='skyblue')
bars2 = ax.bar(x + width/2, r2_values, width, label='R² × 10', color='orange')

ax.set_xlabel('Models', fontsize=10)
ax.set_ylabel('Value', fontsize=10)
ax.set_title('Model Comparison', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(['CatBoost', 'NN', 'Ensemble'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_evaluation_fixed.png', dpi=150, bbox_inches='tight')
print("✅ Plot saved: model_evaluation_fixed.png")


print("\n" + "="*70)
print("🚦 REAL-WORLD SCENARIO TESTING")
print("="*70)

test_scenarios = [
    {"name": "Night (low traffic)", "veh_count": 2, "CO2": 100},
    {"name": "Morning (medium traffic)", "veh_count": 8, "CO2": 500},
    {"name": "Congestion", "veh_count": 15, "CO2": 1200},
    {"name": "Rush hour", "veh_count": 20, "CO2": 1800}
]

for scenario in test_scenarios:
    waiting_time_est = scenario['veh_count'] * 2 
    veh_waiting_ratio = scenario['veh_count'] / (waiting_time_est + 1)
    CO2_per_vehicle = scenario['CO2'] / (scenario['veh_count'] + 1)
    traffic_density = scenario['veh_count'] * waiting_time_est
    
    X_scenario = np.array([[
        scenario['veh_count'],
        scenario['CO2'],
        veh_waiting_ratio,
        CO2_per_vehicle,
        traffic_density
    ]])
    X_scenario_scaled = scaler.transform(X_scenario)
    
    pred_cat = np.clip(cat_model.predict(X_scenario)[0], 5, 90)
    pred_nn = np.clip(nn_model.predict(X_scenario_scaled, verbose=0)[0][0], 5, 90)
    pred_ensemble = (pred_cat + pred_nn) / 2
    
    print(f"\n📍 {scenario['name']}:")
    print(f"   Vehicles: {scenario['veh_count']}, CO2: {scenario['CO2']}g")
    print(f"   → CatBoost:    {pred_cat:.1f}s")
    print(f"   → Neural Net:  {pred_nn:.1f}s")
    print(f"   → Ensemble:    {pred_ensemble:.1f}s ⭐")


print("\n" + "="*70)
print("🎯 FINAL REPORT")
print("="*70)
print(f"✅ Best model: {best_model} (MAE = {best_mae:.2f}s)")
print(f"📊 Overall quality: {'Good' if best_mae < 5 else 'Needs improvement'}")
print(f"⚠️  IMPORTANT: Target variable was created heuristically!")
print(f"   For real-world deployment, experimental data from SUMO is required.")

results_df = pd.DataFrame({
    'y_test': y_test.values,
    'y_pred_catboost': y_pred_cat,
    'y_pred_nn': y_pred_nn,
    'y_pred_ensemble': y_pred_ensemble,
    'abs_error_catboost': np.abs(y_test.values - y_pred_cat),
    'abs_error_nn': np.abs(y_test.values - y_pred_nn),
    'abs_error_ensemble': np.abs(y_test.values - y_pred_ensemble)
})
results_df.to_csv('model_predictions_fixed.csv', index=False)
print("\n✅ Results saved: model_predictions_fixed.csv")
print("="*70)