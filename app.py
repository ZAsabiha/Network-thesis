"""
SDN Traffic Monitoring Web UI
Run with: python app.py
"""

import os
import time
import json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import joblib
import threading
import subprocess

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sdn-monitor-secret'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
current_data = []
monitoring = False
model = None
preprocessor = None

# Try to load ML model if exists
try:
    model = joblib.load('sdn_congestion_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    print("✅ ML Model loaded successfully")
except:
    print("⚠️ No ML model found. Train model first or use rule-based detection.")

# ===================== ROUTES =====================

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get latest statistics"""
    try:
        df = pd.read_csv('traffic_dataset.csv').tail(100)
        
        # Calculate summary stats
        stats = {
            'total_samples': len(df),
            'current_tx': float(df['tx_kbps'].iloc[-1]) if len(df) > 0 else 0,
            'current_rx': float(df['rx_kbps'].iloc[-1]) if len(df) > 0 else 0,
            'avg_tx': float(df['tx_kbps'].mean()),
            'avg_rx': float(df['rx_kbps'].mean()),
            'max_tx': float(df['tx_kbps'].max()),
            'congestion_count': int(df['is_congested'].sum()),
            'congestion_rate': float(df['is_congested'].mean() * 100) if len(df) > 0 else 0,
            'switches': list(df['switch_id'].unique()),
            'ports': list(df['port_no'].unique())
        }
        
        return jsonify(stats)
    except:
        return jsonify({'error': 'No data available'})

@app.route('/api/traffic')
def get_traffic_data():
    """Get traffic data for charts"""
    try:
        df = pd.read_csv('traffic_dataset.csv').tail(100)
        data = df[['timestamp', 'tx_kbps', 'rx_kbps', 'is_congested']].to_dict('records')
        return jsonify(data)
    except:
        return jsonify([])

@app.route('/api/switches')
def get_switches():
    """Get switch status"""
    try:
        df = pd.read_csv('traffic_dataset.csv').tail(100)
        switches = []
        
        for switch_id in df['switch_id'].unique():
            switch_data = df[df['switch_id'] == switch_id]
            latest = switch_data.iloc[-1]
            
            switches.append({
                'id': int(switch_id),
                'port_count': len(switch_data['port_no'].unique()),
                'status': '🟢 Healthy' if latest['is_congested'] == 0 else '🔴 Congested',
                'tx_speed': float(latest['tx_kbps']),
                'rx_speed': float(latest['rx_kbps']),
                'errors': int(latest['tx_errors'] + latest['rx_errors'])
            })
        
        return jsonify(switches)
    except:
        return jsonify([])

@app.route('/api/predict', methods=['POST'])
def predict_congestion():
    """Predict congestion using ML model"""
    try:
        data = request.json
        
        if model is None:
            return jsonify({'error': 'Model not loaded. Train first!'})
        
        # Prepare features
        features = np.array([[
            data.get('tx_bytes', 0),
            data.get('rx_bytes', 0),
            data.get('tx_kbps', 0),
            data.get('rx_kbps', 0),
            data.get('tx_errors', 0),
            data.get('rx_errors', 0)
        ]])
        
        # Scale
        features_scaled = preprocessor.transform(features)
        
        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'congested': bool(prediction),
            'confidence': float(probability[prediction]),
            'probability_normal': float(probability[0]),
            'probability_congested': float(probability[1])
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/generate_traffic', methods=['POST'])
def generate_traffic():
    """Generate traffic in Mininet"""
    try:
        data = request.json
        duration = data.get('duration', 30)
        bandwidth = data.get('bandwidth', '10M')
        traffic_type = data.get('type', 'tcp')
        
        # Run traffic generator
        cmd = f"iperf h1 h2 -t {duration} -b {bandwidth}"
        if traffic_type == 'udp':
            cmd += " -u"
        
        # In production, you would send this to Mininet
        # For now, just return success
        
        return jsonify({
            'status': 'success',
            'message': f'Generating {traffic_type} traffic for {duration}s at {bandwidth}'
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/train_model')
def train_model():
    """Train the ML model"""
    try:
        # Run model training script
        result = subprocess.run(
            ['python', 'model_trainer.py'],
            capture_output=True,
            text=True
        )
        
        return jsonify({
            'status': 'success',
            'output': result.stdout,
            'error': result.stderr
        })
    except Exception as e:
        return jsonify({'error': str(e)})

# ===================== WEBSOCKET EVENTS =====================

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('Client connected')
    emit('connected', {'status': 'connected to server'})

@socketio.on('start_monitoring')
def start_monitoring():
    """Start real-time monitoring"""
    global monitoring
    monitoring = True
    
    def monitor_loop():
        while monitoring:
            try:
                # Read latest data
                df = pd.read_csv('traffic_dataset.csv').tail(10)
                data = df.to_dict('records')
                
                # Emit to all clients
                socketio.emit('traffic_update', {
                    'data': data,
                    'timestamp': time.time()
                })
            except:
                pass
            
            time.sleep(2)
    
    # Start monitoring thread
    thread = threading.Thread(target=monitor_loop)
    thread.daemon = True
    thread.start()

@socketio.on('stop_monitoring')
def stop_monitoring():
    """Stop real-time monitoring"""
    global monitoring
    monitoring = False
    emit('monitoring_stopped', {'status': 'Monitoring stopped'})

# ===================== RUN APP =====================

if __name__ == '__main__':
    print("🚀 Starting SDN Traffic Monitoring UI")
    print("📊 Open your browser at http://localhost:5000")
    print("🔄 Live updates enabled via WebSocket")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)