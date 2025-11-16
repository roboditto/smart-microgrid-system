import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import time
import json

# Page configuration
st.set_page_config(
    page_title="AI Smart Microgrid Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load tier configuration
LOAD_TIERS = {
    1: {'name': 'Critical', 'loads': [0], 'priority': 1, 'color': '#dc3545'},  # Medical, refrigeration, comms
    2: {'name': 'Essential', 'loads': [1], 'priority': 2, 'color': '#ffc107'},  # Lighting, water pumps
    3: {'name': 'Non-Critical', 'loads': [2], 'priority': 3, 'color': '#28a745'}  # Other appliances
}

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-normal {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">⚡ AI-Powered Smart Microgrid Control System</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: gray; margin-top: -1rem;">Renewable Energy • Battery Storage • AI Predictions • Tiered Load Management • Island Mode</p>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("🎛️ System Controls")
st.sidebar.markdown("---")

# Simulation mode toggle
simulation_mode = st.sidebar.checkbox("🔄 Simulation Mode", value=True, help="Enable simulation for testing on Windows")

# Grid connection status
st.sidebar.markdown("### 🔌 Grid Status")
grid_connected = st.sidebar.checkbox("Grid Connected", value=True, help="Toggle island mode operation")
if not grid_connected:
    st.sidebar.warning("⚠️ ISLAND MODE ACTIVE")
    st.sidebar.info("Operating independently from national grid")

# Battery settings
st.sidebar.markdown("### 🔋 Battery Bank Settings")
battery_capacity = st.sidebar.slider("Battery Capacity (kWh)", 10, 100, 50, 5)
battery_min_soc = st.sidebar.slider("Minimum SOC (%)", 10, 50, 20, 5, help="Reserve capacity")

# Power threshold control
power_threshold = st.sidebar.slider(
    "⚠️ Power Threshold (W)",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
    help="Maximum total power consumption before load shedding"
)

# Auto-refresh control
auto_refresh = st.sidebar.checkbox("🔃 Auto-refresh", value=False)
refresh_interval = 3  # Default value
if auto_refresh:
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 1, 10, 3)

# Manual relay controls with tier labels
st.sidebar.markdown("### 🔌 Load Control (Tiered)")
relay_states = []
for i in range(3):
    tier_info = [t for t in LOAD_TIERS.values() if i in t['loads']][0]
    relay_states.append(
        st.sidebar.checkbox(
            f"Load {i+1}: {tier_info['name']} (Tier {tier_info['priority']})",
            value=True,
            help=f"Priority: {tier_info['priority']}"
        )
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🤖 AI Model Settings")
contamination = st.sidebar.slider("Anomaly Detection Sensitivity", 0.01, 0.3, 0.1, 0.01)
enable_predictions = st.sidebar.checkbox("Enable Outage Predictions", value=True)

# Initialize session state for data
if 'data_log' not in st.session_state:
    st.session_state.data_log = []

if 'battery_log' not in st.session_state:
    st.session_state.battery_log = []

if 'solar_log' not in st.session_state:
    st.session_state.solar_log = []

if 'weather_data' not in st.session_state:
    # Simulated weather forecast
    st.session_state.weather_data = {
        'current': 'Sunny',
        'forecast_6h': 'Partly Cloudy',
        'forecast_12h': 'Cloudy',
        'cloud_cover': 20,  # percentage
        'temperature': 28,  # Celsius
        'wind_speed': 15  # km/h
    }

if 'battery_soc' not in st.session_state:
    st.session_state.battery_soc = 75.0  # State of charge percentage

if 'island_mode_activated' not in st.session_state:
    st.session_state.island_mode_activated = False

# Initialize ML models
if 'anomaly_model' not in st.session_state:
    st.session_state.anomaly_model = IsolationForest(contamination=contamination, random_state=42)
    st.session_state.anomaly_model_trained = False

if 'prediction_model' not in st.session_state:
    st.session_state.prediction_model = RandomForestRegressor(n_estimators=100, random_state=42)
    st.session_state.prediction_model_trained = False
    st.session_state.scaler = StandardScaler()

# Load historical training data
if 'historical_data' not in st.session_state:
    try:
        st.session_state.historical_data = pd.read_csv('microgrid_sensor_data.csv')
        st.session_state.historical_data['timestamp'] = pd.to_datetime(st.session_state.historical_data['timestamp'])
        st.session_state.data_loaded = True
    except FileNotFoundError:
        st.session_state.historical_data = None
        st.session_state.data_loaded = False

# Function to get solar production
def get_solar_production(simulation=True, weather_data=None):
    """Get solar panel production data"""
    if simulation:
        hour = datetime.now().hour
        # Solar production curve (peak at midday)
        if 6 <= hour <= 18:
            base_production = 1000 * np.sin(np.pi * (hour - 6) / 12)  # W
            # Weather impact
            cloud_factor = 1 - (weather_data['cloud_cover'] / 100) * 0.7 if weather_data else 1
            production = base_production * cloud_factor
        else:
            production = 0
        
        return {
            'production': round(production + np.random.uniform(-50, 50), 2),
            'voltage': round(48.0 + np.random.uniform(-2, 2), 2),
            'current': round(production / 48.0 if production > 0 else 0, 2)
        }
    else:
        # Hardware reading would go here
        return {'production': 0, 'voltage': 0, 'current': 0}

# Function to simulate battery
def update_battery(current_soc, solar_power, load_power, grid_connected, dt_seconds=1):
    """Update battery state of charge"""
    # Net power (positive = charging, negative = discharging)
    if grid_connected:
        # Grid supplies deficit, battery charges from excess solar
        net_power = solar_power - load_power
    else:
        # Island mode: battery must supply deficit
        net_power = solar_power - load_power
    
    # Convert power to energy (Wh)
    energy_delta = (net_power * dt_seconds) / 3600  # Wh
    
    # Update SOC
    soc_delta = (energy_delta / (battery_capacity * 1000)) * 100  # percentage
    new_soc = np.clip(current_soc + soc_delta, 0, 100)
    
    return new_soc, net_power

# Function to perform intelligent load shedding
def intelligent_load_shedding(loads, battery_soc, solar_power, grid_connected):
    """AI-based load management with priority tiers"""
    total_demand = sum([load['power'] for load in loads if load['state']])
    available_power = solar_power if not grid_connected else solar_power + 5000  # Assume grid can supply 5kW
    
    # Check if we need to shed loads
    if battery_soc < battery_min_soc or total_demand > available_power:
        # Shed loads starting from lowest priority
        recommendations = []
        for tier in sorted(LOAD_TIERS.keys(), reverse=True):
            tier_loads = LOAD_TIERS[tier]['loads']
            for load_idx in tier_loads:
                if load_idx < len(loads) and loads[load_idx]['state']:
                    recommendations.append({
                        'load_id': load_idx + 1,
                        'action': 'shed',
                        'tier': tier,
                        'reason': f"Battery SOC: {battery_soc:.1f}%" if battery_soc < battery_min_soc else "Demand exceeds supply"
                    })
                    total_demand -= loads[load_idx]['power']
                    if total_demand <= available_power and battery_soc >= battery_min_soc:
                        break
            if total_demand <= available_power and battery_soc >= battery_min_soc:
                break
        return recommendations
    return []

# Function to simulate sensor readings
def get_sensor_data(simulation=True):
    """Get sensor data from hardware or simulation"""
    if simulation:
        # Simulate normal operation
        base_voltage = 12.0
        loads = []
        for i, state in enumerate(relay_states):
            if state:
                # Different power levels by tier
                tier_info = [t for t in LOAD_TIERS.values() if i in t['loads']][0]
                if tier_info['priority'] == 1:  # Critical
                    current = np.random.uniform(40, 60)  # Higher consumption
                elif tier_info['priority'] == 2:  # Essential
                    current = np.random.uniform(25, 40)
                else:  # Non-critical
                    current = np.random.uniform(15, 30)
                
                # Occasional anomalies
                if np.random.random() < 0.05:
                    current *= 1.5
                
                voltage = base_voltage + np.random.uniform(-0.5, 0.5)
                power = voltage * current
                loads.append({
                    'load_id': i + 1,
                    'voltage': round(voltage, 2),
                    'current': round(current, 2),
                    'power': round(power, 2),
                    'state': 1 if state else 0,
                    'tier': tier_info['priority']
                })
            else:
                tier_info = [t for t in LOAD_TIERS.values() if i in t['loads']][0]
                loads.append({
                    'load_id': i + 1,
                    'voltage': 0,
                    'current': 0,
                    'power': 0,
                    'state': 0,
                    'tier': tier_info['priority']
                })
        return loads
    else:
        # Import and use actual hardware readings
        try:
            import smartgrid
            loads = []
            for i in range(3):
                current = smartgrid.read_current(i)
                voltage = smartgrid.ina.voltage()
                power = voltage * current
                tier_info = [t for t in LOAD_TIERS.values() if i in t['loads']][0]
                loads.append({
                    'load_id': i + 1,
                    'voltage': round(voltage, 2),
                    'current': round(current, 2),
                    'power': round(power, 2),
                    'state': relay_states[i],
                    'tier': tier_info['priority']
                })
            return loads
        except Exception as e:
            st.error(f"Hardware read error: {e}")
            return []

# Detect island mode transition
if not grid_connected and not st.session_state.island_mode_activated:
    st.session_state.island_mode_activated = True
elif grid_connected:
    st.session_state.island_mode_activated = False

# Get current readings
current_readings = get_sensor_data(simulation_mode)
total_load_power = sum([load['power'] for load in current_readings])

# Get solar production
solar_data = get_solar_production(simulation_mode, st.session_state.weather_data)
solar_power = solar_data['production']

# Update battery
st.session_state.battery_soc, net_battery_power = update_battery(
    st.session_state.battery_soc,
    solar_power,
    total_load_power,
    grid_connected,
    dt_seconds=1
)

# Intelligent load shedding
load_recommendations = intelligent_load_shedding(
    current_readings,
    st.session_state.battery_soc,
    solar_power,
    grid_connected
)

# Island mode alert
if st.session_state.island_mode_activated:
    st.error("🏝️ **ISLAND MODE ACTIVATED** - Operating independently from grid. Essential loads prioritized.")

# Main metrics display
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="☀️ Solar Production",
        value=f"{solar_power:.0f} W",
        delta=f"{solar_power - total_load_power:.0f} W" if solar_power > 0 else "Night"
    )

with col2:
    battery_delta = net_battery_power
    st.metric(
        label="🔋 Battery SOC",
        value=f"{st.session_state.battery_soc:.1f}%",
        delta=f"{'Charging' if battery_delta > 0 else 'Discharging' if battery_delta < 0 else 'Idle'}"
    )

with col3:
    st.metric(
        label="⚡ Load Demand",
        value=f"{total_load_power:.0f} W",
        delta=f"{total_load_power - power_threshold:.0f} W from threshold"
    )

with col4:
    active_loads = sum([1 for load in current_readings if load['state'] == 1])
    st.metric(
        label="🔌 Active Loads",
        value=f"{active_loads}/3",
    )

with col5:
    if grid_connected:
        system_status = "✅ GRID"
        if total_load_power > power_threshold:
            system_status = "⚠️ HIGH LOAD"
    else:
        system_status = "🏝️ ISLAND"
    st.metric(
        label="System Status",
        value=system_status
    )

# Add current data to logs
timestamp = datetime.now()
for load in current_readings:
    st.session_state.data_log.append({
        'timestamp': timestamp,
        'load_id': load['load_id'],
        'voltage': load['voltage'],
        'current': load['current'],
        'power': load['power'],
        'state': load['state'],
        'tier': load['tier']
    })

st.session_state.battery_log.append({
    'timestamp': timestamp,
    'soc': st.session_state.battery_soc,
    'power': net_battery_power,
    'capacity_kwh': battery_capacity
})

st.session_state.solar_log.append({
    'timestamp': timestamp,
    'production': solar_power,
    'voltage': solar_data['voltage'],
    'current': solar_data['current']
})

# Keep only last 100 readings per load
if len(st.session_state.data_log) > 300:
    st.session_state.data_log = st.session_state.data_log[-300:]
if len(st.session_state.battery_log) > 100:
    st.session_state.battery_log = st.session_state.battery_log[-100:]
if len(st.session_state.solar_log) > 100:
    st.session_state.solar_log = st.session_state.solar_log[-100:]

# Create DataFrame
df = pd.DataFrame(st.session_state.data_log)

# Show load shedding recommendations
if len(load_recommendations) > 0:
    st.warning(f"⚠️ **AI Load Management Alert**: {len(load_recommendations)} load(s) recommended for shedding")
    rec_cols = st.columns(len(load_recommendations))
    for idx, rec in enumerate(load_recommendations):
        with rec_cols[idx]:
            st.error(f"**Load {rec['load_id']}** (Tier {rec['tier']})\n\n{rec['reason']}")

# Renewable Energy Overview
st.markdown("## ☀️ Renewable Energy Input")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Solar Production & Battery Status")
    if len(st.session_state.solar_log) > 0 and len(st.session_state.battery_log) > 0:
        solar_df = pd.DataFrame(st.session_state.solar_log)
        battery_df = pd.DataFrame(st.session_state.battery_log)
        
        fig = go.Figure()
        
        # Solar production
        fig.add_trace(go.Scatter(
            x=solar_df['timestamp'],
            y=solar_df['production'],
            name='Solar Production',
            line=dict(color='orange', width=2),
            fill='tozeroy'
        ))
        
        # Battery SOC on secondary axis
        fig.add_trace(go.Scatter(
            x=battery_df['timestamp'],
            y=battery_df['soc'],
            name='Battery SOC',
            line=dict(color='green', width=2),
            yaxis='y2'
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Solar Power (W)",
            yaxis2=dict(
                title="Battery SOC (%)",
                overlaying='y',
                side='right',
                range=[0, 100]
            ),
            hovermode='x unified',
            height=350
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### ☁️ Weather Forecast")
    
    weather = st.session_state.weather_data
    st.info(f"**Current:** {weather['current']}")
    st.info(f"**6h:** {weather['forecast_6h']}")
    st.info(f"**12h:** {weather['forecast_12h']}")
    
    st.metric("Cloud Cover", f"{weather['cloud_cover']}%")
    st.metric("Temperature", f"{weather['temperature']}°C")
    st.metric("Wind Speed", f"{weather['wind_speed']} km/h")
    
    # Battery capacity indicator
    st.markdown("### 🔋 Battery Capacity")
    battery_energy = (st.session_state.battery_soc / 100) * battery_capacity
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=st.session_state.battery_soc,
        title={'text': f"{battery_energy:.1f} kWh / {battery_capacity} kWh"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "green" if st.session_state.battery_soc > 50 else "orange" if st.session_state.battery_soc > 20 else "red"},
            'steps': [
                {'range': [0, 20], 'color': "lightgray"},
                {'range': [20, 50], 'color': "lightyellow"},
                {'range': [50, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': battery_min_soc
            }
        }
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

# Real-time monitoring section
st.markdown("## 📈 Load Management Layer (Tiered Priority System)")

# Power consumption chart
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Power Consumption by Load")
    
    if len(df) > 0:
        # Line chart showing power over time
        fig = go.Figure()
        
        for load_id in [1, 2, 3]:
            load_data = df[df['load_id'] == load_id].tail(50)
            fig.add_trace(go.Scatter(
                x=load_data['timestamp'],
                y=load_data['power'],
                mode='lines+markers',
                name=f'Load {load_id}',
                line=dict(width=2)
            ))
        
        # Add threshold line
        fig.add_hline(
            y=power_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Threshold",
            annotation_position="right"
        )
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Power (W)",
            hovermode='x unified',
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Tiered Load Status")
    
    # Create gauge charts for each load with tier info
    for load in current_readings:
        tier_info = [t for t in LOAD_TIERS.values() if load['tier'] == t['priority']][0]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=load['power'],
            title={'text': f"Load {load['load_id']}: {tier_info['name']}<br>Tier {tier_info['priority']}"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': tier_info['color'] if load['state'] else "gray"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': power_threshold / 3
                }
            }
        ))
        fig.update_layout(height=150, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    
    # Tier legend
    st.markdown("**Priority Tiers:**")
    st.markdown("🔴 Tier 1: Critical (Medical, Refrigeration, Comms)")
    st.markdown("🟡 Tier 2: Essential (Lighting, Water Pumps)")
    st.markdown("🟢 Tier 3: Non-Critical (Other Appliances)")

# Machine Learning Analysis Section
st.markdown("## 🤖 Smart AI Energy Controller")
st.markdown("*Monitoring: Weather • Solar Production • Battery Capacity • Electricity Demand*")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Anomaly Detection Status")
    
    # Train on historical data if available
    if st.session_state.data_loaded and st.session_state.historical_data is not None:
        hist_df = st.session_state.historical_data
        
        # Prepare features from historical data
        feature_cols = ['ina219_voltage', 'acs712_current', 'power_watts', 'load_state', 
                       'solar_production', 'battery_soc', 'hour']
        X_hist = hist_df[feature_cols].values
        
        # Train anomaly detection model
        if not st.session_state.anomaly_model_trained:
            st.session_state.anomaly_model.fit(X_hist)
            st.session_state.anomaly_model_trained = True
        
        st.success(f"✅ Model Trained on {len(hist_df):,} historical samples")
        
        # Show historical anomaly stats
        hist_anomalies = len(hist_df[hist_df['classification'] == 'ANOMALY'])
        st.metric("Historical Anomalies", f"{hist_anomalies} ({hist_anomalies/len(hist_df)*100:.2f}%)")
        
        # Predict on current data if available
        if len(df) > 0:
            # Match feature columns for current data
            current_features = []
            for _, row in df.tail(1).iterrows():
                current_features.append([
                    row['voltage'],
                    row['current'], 
                    row['power'],
                    row['state'],
                    solar_power,
                    st.session_state.battery_soc,
                    datetime.now().hour
                ])
            
            prediction = st.session_state.anomaly_model.predict(current_features)
            latest_prediction = "🚨 ANOMALY DETECTED" if prediction[0] == -1 else "✅ Normal Operation"
            st.info(f"Latest Reading: {latest_prediction}")
    
    elif len(df) > 10:
        # Fallback to live data only
        feature_cols = ['voltage', 'current', 'power', 'state']
        X = df[feature_cols].values
        
        # Train/update anomaly detection model
        st.session_state.anomaly_model.fit(X)
        st.session_state.anomaly_model_trained = True
        
        # Predict anomalies
        predictions = st.session_state.anomaly_model.predict(X)
        anomaly_scores = st.session_state.anomaly_model.score_samples(X)
        
        # Add predictions to dataframe
        df['anomaly'] = predictions
        df['anomaly_score'] = anomaly_scores
        
        # Count anomalies
        n_anomalies = len(df[df['anomaly'] == -1])
        anomaly_rate = (n_anomalies / len(df)) * 100
        
        st.success(f"✅ Model Trained on {len(df)} live samples")
        st.metric("Detected Anomalies", f"{n_anomalies} ({anomaly_rate:.1f}%)")
        
        # Latest prediction
        latest_prediction = "🚨 ANOMALY DETECTED" if predictions[-1] == -1 else "✅ Normal Operation"
        st.info(f"Latest Reading: {latest_prediction}")
        
    else:
        st.warning("⏳ Collecting data... Need at least 10 samples to train model")
        if not st.session_state.data_loaded:
            st.info("💡 Run `generate_training_data.py` to create historical dataset")

with col2:
    st.markdown("### Outage Prediction & Energy Forecast")
    
    if enable_predictions and st.session_state.data_loaded and st.session_state.historical_data is not None:
        hist_df = st.session_state.historical_data
        
        # Train predictive model on historical data
        if not st.session_state.prediction_model_trained:
            # Features for prediction
            pred_features = ['hour', 'solar_production', 'total_load_power', 'battery_soc', 'cloud_cover']
            X_train = hist_df[pred_features].values
            y_train = hist_df['battery_soc'].shift(-12).ffill().to_numpy()  # Predict 1 hour ahead
            
            # Scale features
            X_train_scaled = st.session_state.scaler.fit_transform(X_train)
            
            # Train model
            st.session_state.prediction_model.fit(X_train_scaled, y_train)
            st.session_state.prediction_model_trained = True
        
        # Make predictions
        import generate_training_data
        current_hour = datetime.now().hour
        future_predictions = []
        
        for hours_ahead in range(1, 7):
            future_hour = (current_hour + hours_ahead) % 24
            
            # Estimate future solar (simplified)
            future_solar = get_solar_production(simulation_mode, st.session_state.weather_data)['production'] if simulation_mode else solar_power
            
            # Predict features
            pred_input = np.array([[
                future_hour,
                future_solar,
                total_load_power,
                st.session_state.battery_soc if hours_ahead == 1 else future_predictions[-1],
                st.session_state.weather_data['cloud_cover']
            ]])
            
            pred_input_scaled = st.session_state.scaler.transform(pred_input)
            predicted_soc = st.session_state.prediction_model.predict(pred_input_scaled)[0]
            future_predictions.append(np.clip(predicted_soc, 0, 100))
        
        # Outage risk assessment
        min_predicted_soc = min(future_predictions)
        outage_risk = "Low"
        risk_color = "green"
        
        if min_predicted_soc < battery_min_soc:
            outage_risk = "High"
            risk_color = "red"
        elif min_predicted_soc < 40:
            outage_risk = "Medium"
            risk_color = "orange"
        
        st.metric(
            "Outage Risk (Next 6h)",
            outage_risk,
            delta=f"Predicted min SOC: {min_predicted_soc:.1f}%"
        )
        
        if outage_risk == "High":
            st.error("⚠️ **Warning**: High risk of power shortage. Consider reducing non-critical loads.")
        elif outage_risk == "Medium":
            st.warning("⚡ **Caution**: Battery running low. Monitor closely.")
        else:
            st.success("✅ **Stable**: Sufficient power reserves.")
        
        # Energy forecast chart
        future_times = [datetime.now() + timedelta(hours=i) for i in range(1, 7)]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=future_times,
            y=future_predictions,
            mode='lines+markers',
            name='AI Predicted SOC',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_hline(
            y=battery_min_soc,
            line_dash="dot",
            line_color="red",
            annotation_text="Min SOC",
            annotation_position="right"
        )
        
        fig.update_layout(
            title="AI-Powered 6-Hour Battery Forecast",
            xaxis_title="Time",
            yaxis_title="Battery SOC (%)",
            height=250,
            yaxis_range=[0, 100]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif enable_predictions and len(df) > 10:
        # Prepare features for prediction
        if len(st.session_state.solar_log) > 10 and len(st.session_state.battery_log) > 10:
            solar_df = pd.DataFrame(st.session_state.solar_log)
            battery_df = pd.DataFrame(st.session_state.battery_log)
            
            # Create training data (simplified)
            recent_solar = solar_df['production'].tail(20).to_numpy()
            recent_battery = battery_df['soc'].tail(20).to_numpy()
            
            # Predict next hour battery SOC
            avg_solar_trend = np.mean(np.diff(recent_solar[-10:])) if len(recent_solar) > 10 else 0
            avg_battery_trend = np.mean(np.diff(recent_battery[-10:])) if len(recent_battery) > 10 else 0
            
            predicted_soc_1h = st.session_state.battery_soc + (avg_battery_trend * 60)
            predicted_soc_1h = np.clip(predicted_soc_1h, 0, 100)
            
            # Outage risk assessment
            outage_risk = "Low"
            risk_color = "green"
            if predicted_soc_1h < battery_min_soc and solar_power < total_load_power:
                outage_risk = "High"
                risk_color = "red"
            elif predicted_soc_1h < 40:
                outage_risk = "Medium"
                risk_color = "orange"
            
            st.metric(
                "Outage Risk (Next Hour)",
                outage_risk,
                delta=f"Predicted SOC: {predicted_soc_1h:.1f}%"
            )
            
            if outage_risk == "High":
                st.error("⚠️ **Warning**: High risk of power shortage. Consider reducing non-critical loads.")
            elif outage_risk == "Medium":
                st.warning("⚡ **Caution**: Battery running low. Monitor closely.")
            else:
                st.success("✅ **Stable**: Sufficient power reserves.")
            
            # Energy forecast chart
            future_times = [datetime.now() + timedelta(hours=i) for i in range(1, 7)]
            forecast_soc = [predicted_soc_1h]
            current_soc_forecast = predicted_soc_1h
            
            for _ in range(5):
                current_soc_forecast += avg_battery_trend * 60 * 0.8  # Damping
                current_soc_forecast = np.clip(current_soc_forecast, 0, 100)
                forecast_soc.append(current_soc_forecast)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=future_times,
                y=forecast_soc,
                mode='lines+markers',
                name='Predicted SOC',
                line=dict(color='blue', dash='dash')
            ))
            
            fig.add_hline(
                y=battery_min_soc,
                line_dash="dot",
                line_color="red",
                annotation_text="Min SOC",
                annotation_position="right"
            )
            
            fig.update_layout(
                title="6-Hour Battery SOC Forecast",
                xaxis_title="Time",
                yaxis_title="Battery SOC (%)",
                height=250,
                yaxis_range=[0, 100]
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Anomaly Score Timeline")
    
    if len(df) > 10 and 'anomaly_score' in df.columns:
        recent_df = df.tail(50)
        
        fig = go.Figure()
        
        # Normal points
        normal_data = recent_df[recent_df['anomaly'] == 1]
        fig.add_trace(go.Scatter(
            x=normal_data['timestamp'],
            y=normal_data['anomaly_score'],
            mode='markers',
            name='Normal',
            marker=dict(color='green', size=8)
        ))
        
        # Anomaly points
        anomaly_data = recent_df[recent_df['anomaly'] == -1]
        fig.add_trace(go.Scatter(
            x=anomaly_data['timestamp'],
            y=anomaly_data['anomaly_score'],
            mode='markers',
            name='Anomaly',
            marker=dict(color='red', size=12, symbol='x')
        ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Historical Data Analysis
st.markdown("## 📜 Historical Data Analysis")

if st.session_state.data_loaded and st.session_state.historical_data is not None:
    historical_df = st.session_state.historical_data
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📊 Sensor Data Overview")
        
        # Classification distribution
        classification_counts = historical_df['classification'].value_counts()
        fig = px.pie(
            values=classification_counts.values,
            names=classification_counts.index,
            title="Normal vs Anomaly Events",
            color_discrete_map={'NORMAL': 'green', 'ANOMALY': 'red'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("### ⚡ Power Consumption Patterns")
        
        # Average power by tier
        tier_power = historical_df.groupby('tier')['power_watts'].mean()
        fig = px.bar(
            x=['Tier 1 (Critical)', 'Tier 2 (Essential)', 'Tier 3 (Non-Critical)'],
            y=tier_power.values,
            title="Average Power by Load Tier",
            labels={'x': 'Load Tier', 'y': 'Power (W)'},
            color=tier_power.values,
            color_continuous_scale='reds'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with col3:
        st.markdown("### 📈 Time-based Patterns")
        
        # Power by hour of day
        hourly_power = historical_df.groupby('hour')['power_watts'].mean()
        fig = px.line(
            x=hourly_power.index,
            y=hourly_power.values,
            title="Average Load by Hour",
            labels={'x': 'Hour of Day', 'y': 'Power (W)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed sensor statistics
    st.markdown("### 🔬 Sensor Statistics (ACS712 & INA219)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**INA219 Voltage Sensor**")
        st.write(f"Mean: {historical_df['ina219_voltage'].mean():.2f} V")
        st.write(f"Min: {historical_df['ina219_voltage'].min():.2f} V")
        st.write(f"Max: {historical_df['ina219_voltage'].max():.2f} V")
        st.write(f"Std Dev: {historical_df['ina219_voltage'].std():.2f} V")
    
    with col2:
        st.markdown("**ACS712 Current Sensor**")
        st.write(f"Mean: {historical_df['acs712_current'].mean():.3f} A")
        st.write(f"Min: {historical_df['acs712_current'].min():.3f} A")
        st.write(f"Max: {historical_df['acs712_current'].max():.3f} A")
        st.write(f"ADC Range: {historical_df['acs712_adc'].min()}-{historical_df['acs712_adc'].max()}")
    
    with col3:
        st.markdown("**Power Measurements**")
        st.write(f"Mean: {historical_df['power_watts'].mean():.2f} W")
        st.write(f"Peak: {historical_df['power_watts'].max():.2f} W")
        st.write(f"Total Samples: {len(historical_df):,}")
        st.write(f"Date Range: {(historical_df['timestamp'].max() - historical_df['timestamp'].min()).days} days")
    
    # Solar and battery trends
    st.markdown("### ☀️ Solar Production & Battery Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Solar production by hour
        solar_by_hour = historical_df.groupby('hour')['solar_production'].mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=solar_by_hour.index,
            y=solar_by_hour.values,
            fill='tozeroy',
            name='Solar Production',
            line=dict(color='orange')
        ))
        fig.update_layout(
            title="Average Solar Production by Hour",
            xaxis_title="Hour of Day",
            yaxis_title="Power (W)",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Battery SOC distribution
        fig = px.histogram(
            historical_df,
            x='battery_soc',
            nbins=30,
            title="Battery SOC Distribution",
            labels={'battery_soc': 'State of Charge (%)'},
            color_discrete_sequence=['green']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent anomalies
    st.markdown("### 🚨 Recent Anomalies")
    anomaly_events = historical_df[historical_df['classification'] == 'ANOMALY'].tail(20)
    
    if len(anomaly_events) > 0:
        st.dataframe(
            anomaly_events[['timestamp', 'tier', 'ina219_voltage', 'acs712_current', 'power_watts', 'battery_soc']],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("No anomalies detected in dataset!")

else:
    # Fallback to old CSV if new data not available
    try:
        historical_df = pd.read_csv('alerts_scoreboard.csv', 
                                     names=['timestamp', 'features', 'classification', 'score'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Classification Distribution")
            
            classification_counts = historical_df['classification'].value_counts()
            fig = px.pie(
                values=classification_counts.values,
                names=classification_counts.index,
                title="Normal vs Malicious Events",
                color_discrete_map={'Normal': 'green', 'MALICIOUS': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown("### Anomaly Score Distribution")
            
            fig = px.histogram(
                historical_df,
                x='score',
                color='classification',
                title="Score Distribution by Classification",
                nbins=30,
                color_discrete_map={'Normal': 'green', 'MALICIOUS': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent alerts
        st.markdown("### 🚨 Recent Alerts")
        malicious_events = historical_df[historical_df['classification'] == 'MALICIOUS'].tail(10)
        
        if len(malicious_events) > 0:
            st.dataframe(
                malicious_events[['timestamp', 'classification', 'score']],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No malicious events detected in recent history!")
            
    except FileNotFoundError:
        st.warning("⚠️ No historical data files found. Run `generate_training_data.py` to create sensor dataset.")

# System summary table
st.markdown("## 📋 System Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### ☀️ Solar System")
    if len(st.session_state.solar_log) > 0:
        latest_solar = st.session_state.solar_log[-1]
        st.write(f"**Production:** {latest_solar['production']:.0f} W")
        st.write(f"**Voltage:** {latest_solar['voltage']:.1f} V")
        st.write(f"**Current:** {latest_solar['current']:.1f} A")

with col2:
    st.markdown("### 🔋 Battery Bank")
    if len(st.session_state.battery_log) > 0:
        latest_battery = st.session_state.battery_log[-1]
        st.write(f"**SOC:** {latest_battery['soc']:.1f}%")
        st.write(f"**Power:** {latest_battery['power']:.0f} W")
        st.write(f"**Capacity:** {battery_capacity} kWh")
        battery_health = "Excellent" if latest_battery['soc'] > 70 else "Good" if latest_battery['soc'] > 40 else "Low"
        st.write(f"**Status:** {battery_health}")

with col3:
    st.markdown("### ⚡ Load Summary")
    st.write(f"**Total Demand:** {total_load_power:.0f} W")
    st.write(f"**Active Loads:** {active_loads}/3")
    tier_1_active = sum([1 for l in current_readings if l['tier'] == 1 and l['state']])
    tier_2_active = sum([1 for l in current_readings if l['tier'] == 2 and l['state']])
    tier_3_active = sum([1 for l in current_readings if l['tier'] == 3 and l['state']])
    st.write(f"**Tier 1 (Critical):** {tier_1_active} active")
    st.write(f"**Tier 2 (Essential):** {tier_2_active} active")
    st.write(f"**Tier 3 (Non-Critical):** {tier_3_active} active")

st.markdown("### Recent Sensor Readings")
if len(df) > 0:
    st.dataframe(
        df.tail(20)[['timestamp', 'load_id', 'voltage', 'current', 'power', 'state', 'tier']],
        use_container_width=True,
        hide_index=True
    )

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
        AI-Powered Smart Microgrid System v2.0<br>
        ☀️ Renewable Energy • 🔋 Battery Storage • 🤖 AI Predictions • 📊 Tiered Load Management • {'🏝️ Island Mode' if not grid_connected else '🔌 Grid Connected'}<br>
        <small>Ensuring essential loads remain powered during outages</small>
    </div>
    """,
    unsafe_allow_html=True
)
