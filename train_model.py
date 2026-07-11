"""
Train SDN Traffic Congestion Model
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Model Training...")

# Check if data exists
try:
    df = pd.read_csv('traffic_dataset.csv')
    print(f"✅ Loaded {len(df)} samples from traffic_dataset.csv")
except FileNotFoundError:
    print("❌ traffic_dataset.csv not found!")
    print("Please generate traffic in Mininet first.")
    exit()

# Check if we have enough data
if len(df) < 10:
    print(f"⚠️ Only {len(df)} samples found. Please generate more traffic!")
    exit()

# Prepare features
feature_cols = ['tx_bytes', 'rx_bytes', 'tx_kbps', 'rx_kbps', 'tx_errors', 'rx_errors']

# Check if all columns exist
missing_cols = [col for col in feature_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Missing columns: {missing_cols}")
    print(f"Available columns: {df.columns.tolist()}")
    exit()

X = df[feature_cols].copy()
y = df['is_congested'].copy()

# Handle missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training samples: {len(X_train)}")
print(f"📊 Test samples: {len(X_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("🌲 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy: {accuracy*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Congested']))

# Feature importance
print("\n📈 Top 5 Important Features:")
importance = model.feature_importances_
for i, (feature, imp) in enumerate(sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)[:5]):
    print(f"   {i+1}. {feature}: {imp*100:.1f}%")

# Save model and preprocessor
joblib.dump(model, 'sdn_congestion_model.pkl')
joblib.dump(scaler, 'preprocessor.pkl')

print("\n💾 Model saved as 'sdn_congestion_model.pkl'")
print("💾 Preprocessor saved as 'preprocessor.pkl'")

# Test prediction on latest data
print("\n🔮 Testing on latest data sample:")
latest = df.iloc[-1]
print(f"   Switch: {latest['switch_id']}, Port: {latest['port_no']}")
print(f"   TX Speed: {latest['tx_kbps']:.2f} Kbps")
print(f"   Actual Congestion: {'🔴 Yes' if latest['is_congested'] == 1 else '🟢 No'}")

# Make prediction
features = [[
    latest['tx_bytes'],
    latest['rx_bytes'],
    latest['tx_kbps'],
    latest['rx_kbps'],
    latest['tx_errors'],
    latest['rx_errors']
]]
features_scaled = scaler.transform(features)
pred = model.predict(features_scaled)[0]
prob = model.predict_proba(features_scaled)[0]

print(f"   Predicted: {'🔴 Congested' if pred == 1 else '🟢 Normal'}")
print(f"   Confidence: {max(prob)*100:.1f}%")

print("\n🎉 Training complete!")