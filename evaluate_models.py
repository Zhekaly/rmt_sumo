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

print("📂 Загрузка данных...")
df = pd.read_csv("traffic_data.csv")

print(f"✓ Загружено {len(df)} записей")
print(f"✓ Колонки: {df.columns.tolist()}")
print(f"\n📊 Статистика данных:")
print(df.describe())

if df.isnull().sum().sum() > 0:
    print("\n⚠️  Обнаружены пропуски:")
    print(df.isnull().sum())
    df = df.dropna()
    print(f"✓ После удаления: {len(df)} записей")


print("\n🔧 Подготовка признаков...")

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
    Эвристика для оптимального времени зелёного света
    В реальности эти значения должны быть получены из экспериментов
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

print(f"Признаки: {X.columns.tolist()}")
print(f"Целевая переменная: оптимальное время зелёного (5-90 сек)")
print(f"Среднее: {y.mean():.1f}s, Медиана: {y.median():.1f}s")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

print(f"\n📊 Разделение:")
print(f"   Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"   Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print("\n🤖 Обучение моделей...")

print("   → CatBoost (с early stopping)...")
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

print("   → Neural Network (с early stopping)...")
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

print("✅ Обучение завершено!")
print(f"   CatBoost: {cat_model.tree_count_} деревьев")
print(f"   NN: остановка на эпохе {len(history.history['loss'])}")

print("\n🔮 Получение предсказаний...")

y_pred_cat = cat_model.predict(X_test)
y_pred_nn = nn_model.predict(X_test_scaled, verbose=0).flatten()
y_pred_ensemble = (y_pred_cat + y_pred_nn) / 2

y_pred_cat = np.clip(y_pred_cat, 5, 90)
y_pred_nn = np.clip(y_pred_nn, 5, 90)
y_pred_ensemble = np.clip(y_pred_ensemble, 5, 90)

print("\n" + "="*70)
print("📈 МЕТРИКИ КАЧЕСТВА")
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
    
    print(f"   MAE:           {mae:.2f}s  (средняя ошибка)")
    print(f"   RMSE:          {rmse:.2f}s  (штраф за большие ошибки)")
    print(f"   R²:            {r2:.4f}  (качество объяснения: {r2*100:.1f}%)")
    print(f"   MAPE:          {mape:.2f}%")
    print(f"   Точность ±5s:  {within_5s:.1f}%")
    print(f"   Точность ±10s: {within_10s:.1f}%")
    
    if mae < best_mae:
        best_mae = mae
        best_model = model_name


print("\n" + "="*70)
print("🔍 ВАЖНОСТЬ ПРИЗНАКОВ (CatBoost)")
print("="*70)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': cat_model.feature_importances_
}).sort_values('importance', ascending=False)

for _, row in feature_importance.iterrows():
    print(f"   {row['feature']:20s}: {row['importance']:6.2f}")

print("\n📊 Создание графиков...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Оценка качества моделей светофора', fontsize=16, fontweight='bold')

ax = axes[0, 0]
ax.scatter(y_test, y_pred_cat, alpha=0.6, s=30, edgecolors='k', linewidths=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Реальное время (сек)', fontsize=10)
ax.set_ylabel('Предсказанное время (сек)', fontsize=10)
ax.set_title(f'CatBoost (MAE={mean_absolute_error(y_test, y_pred_cat):.2f}s)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.scatter(y_test, y_pred_nn, alpha=0.6, s=30, color='green', edgecolors='k', linewidths=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Реальное время (сек)', fontsize=10)
ax.set_ylabel('Предсказанное время (сек)', fontsize=10)
ax.set_title(f'Neural Network (MAE={mean_absolute_error(y_test, y_pred_nn):.2f}s)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(y_test, y_pred_ensemble, alpha=0.6, s=30, color='purple', edgecolors='k', linewidths=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Реальное время (сек)', fontsize=10)
ax.set_ylabel('Предсказанное время (сек)', fontsize=10)
ax.set_title(f'Ensemble (MAE={mean_absolute_error(y_test, y_pred_ensemble):.2f}s)', fontsize=11)
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
errors_cat = y_test.values - y_pred_cat
errors_nn = y_test.values - y_pred_nn
ax.hist(errors_cat, bins=30, alpha=0.6, label='CatBoost', color='blue')
ax.hist(errors_nn, bins=30, alpha=0.6, label='Neural Net', color='green')
ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Ошибка (сек)', fontsize=10)
ax.set_ylabel('Частота', fontsize=10)
ax.set_title('Распределение ошибок', fontsize=11)
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
feature_importance.plot(kind='barh', x='feature', y='importance', ax=ax, legend=False, color='teal')
ax.set_xlabel('Важность', fontsize=10)
ax.set_ylabel('')
ax.set_title('Важность признаков (CatBoost)', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')

ax = axes[1, 2]
mae_values = [mean_absolute_error(y_test, p) for p in [y_pred_cat, y_pred_nn, y_pred_ensemble]]
r2_values = [r2_score(y_test, p) * 10 for p in [y_pred_cat, y_pred_nn, y_pred_ensemble]]

x = np.arange(3)
width = 0.35

bars1 = ax.bar(x - width/2, mae_values, width, label='MAE (сек)', color='skyblue')
bars2 = ax.bar(x + width/2, r2_values, width, label='R² × 10', color='orange')

ax.set_xlabel('Модели', fontsize=10)
ax.set_ylabel('Значение', fontsize=10)
ax.set_title('Сравнение моделей', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(['CatBoost', 'NN', 'Ensemble'])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('model_evaluation_fixed.png', dpi=150, bbox_inches='tight')
print("✅ График сохранён: model_evaluation_fixed.png")


print("\n" + "="*70)
print("🚦 ТЕСТ НА РЕАЛЬНЫХ СЦЕНАРИЯХ")
print("="*70)

test_scenarios = [
    {"name": "Ночь (малый трафик)", "veh_count": 2, "CO2": 100},
    {"name": "Утро (средний трафик)", "veh_count": 8, "CO2": 500},
    {"name": "Пробка", "veh_count": 15, "CO2": 1200},
    {"name": "Час пик", "veh_count": 20, "CO2": 1800}
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
    print(f"   Машины: {scenario['veh_count']}, CO2: {scenario['CO2']}g")
    print(f"   → CatBoost:    {pred_cat:.1f}s")
    print(f"   → Neural Net:  {pred_nn:.1f}s")
    print(f"   → Ансамбль:    {pred_ensemble:.1f}s ⭐")


print("\n" + "="*70)
print("🎯 ИТОГОВЫЙ ОТЧЁТ")
print("="*70)
print(f"✅ Лучшая модель: {best_model} (MAE = {best_mae:.2f}s)")
print(f"📊 Общее качество: {'Хорошее' if best_mae < 5 else 'Требует улучшения'}")
print(f"⚠️  ВАЖНО: Целевая переменная создана эвристически!")
print(f"   Для реального применения нужны данные из экспериментов с SUMO.")

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
print("\n✅ Результаты сохранены: model_predictions_fixed.csv")
print("="*70)