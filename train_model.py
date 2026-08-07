import xgboost as xgb
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_engineering import build_features
from features import FEATURES, TARGET   # ✅ FIX

df = build_features()

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror"
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

rmse = mean_squared_error(y_test, preds) ** 0.5

print("RMSE:", rmse)

joblib.dump(model, "xgb_model.pkl")

print("Model saved successfully")