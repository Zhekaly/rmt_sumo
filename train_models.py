import pandas as pd
from catboost import CatBoostRegressor
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("traffic_data.csv")
X = df[["veh_count", "waiting_time", "CO2"]]
y = df["waiting_time"].apply(lambda x: max(5, min(60, x / 2)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=4, verbose=False)
cat_model.fit(X_train, y_train)
cat_model.save_model("catboost_model.cbm")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

nn_model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(3,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_scaled, y_train, epochs=10, verbose=1)
nn_model.save("nn_model.keras")

joblib.dump(scaler, "scaler.pkl")
print("Models trained and saved (catboost_model.cbm, nn_model.keras, scaler.pkl)")
