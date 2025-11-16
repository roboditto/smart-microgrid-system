"""
Generate realistic training data for smart microgrid predictive analytics
Simulates ACS712 current sensor and INA219 voltage/power sensor readings
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration
NUM_DAYS = 7  # Generate 7 days of data
SAMPLES_PER_HOUR = 12  # One sample every 5 minutes
TOTAL_SAMPLES = NUM_DAYS * 24 * SAMPLES_PER_HOUR

# ACS712 specifications (5A version)
# Sensitivity: 185 mV/A
# Zero current output: 2.5V (at 5V supply)
# With 3.3V supply: zero current at ~1.65V
# Output range: 0-3.3V
# Current range: -5A to +5A

# INA219 specifications
# Voltage range: 0-26V (typically 12V system)
# Current measurement via shunt resistor (0.1Ω typical)
# Power calculation: V * I

def generate_solar_production(hour, cloud_cover=0.2):
    """Generate realistic solar production based on time of day"""
    if 6 <= hour <= 18:
        # Peak production at solar noon (12:00)
        peak_power = 1000  # Watts
        hour_angle = (hour - 12) / 6  # Normalize to -1 to 1
        base_production = peak_power * np.cos(hour_angle * np.pi / 2)
        
        # Apply cloud cover effect
        cloud_factor = 1 - (cloud_cover * 0.7)
        production = max(0, base_production * cloud_factor)
        
        # Add some random variation
        production += np.random.normal(0, 50)
        return max(0, production)
    else:
        return 0  # No production at night

def generate_load_current(tier, hour, day_of_week, anomaly=False):
    """Generate realistic current draw for different load tiers"""
    # Base current ranges by tier (Amperes)
    tier_ranges = {
        1: (3.0, 5.0),    # Critical: Medical equipment, refrigeration (35-60W @ 12V)
        2: (2.0, 3.5),    # Essential: Lighting, pumps (24-42W @ 12V)
        3: (1.2, 2.5)     # Non-critical: Appliances (14-30W @ 12V)
    }
    
    # Time-based usage patterns
    if 22 <= hour or hour <= 5:  # Night
        usage_factor = 0.4 if tier == 1 else 0.2  # Critical loads stay on
    elif 6 <= hour <= 8 or 17 <= hour <= 21:  # Morning/evening peaks
        usage_factor = 0.9
    else:  # Daytime
        usage_factor = 0.6
    
    # Weekend variation
    if day_of_week >= 5:  # Weekend
        usage_factor *= 0.8
    
    min_current, max_current = tier_ranges[tier]
    base_current = np.random.uniform(min_current, max_current) * usage_factor
    
    # Anomaly simulation (power surge, fault)
    if anomaly:
        base_current *= np.random.uniform(2.0, 3.5)
    
    # Add noise
    current = base_current + np.random.normal(0, 0.1)
    
    # Clamp to ACS712 range
    return np.clip(current, 0, 5.0)

def current_to_acs712_voltage(current):
    """Convert current to ACS712 sensor output voltage (3.3V supply)"""
    # Zero current: 1.65V, Sensitivity: 185mV/A (for 5V supply)
    # Scaled to 3.3V: zero at 1.089V, sensitivity ~122mV/A
    zero_voltage = 1.65
    sensitivity = 0.185 * (3.3 / 5.0)  # Scale sensitivity for 3.3V
    voltage = zero_voltage + (current * sensitivity)
    return np.clip(voltage, 0, 3.3)

def current_to_adc(voltage):
    """Convert voltage to ADC reading (10-bit ADC, 0-1023)"""
    # Assuming 3.3V reference
    adc_value = (voltage / 3.3) * 1023
    return int(np.clip(adc_value, 0, 1023))

def generate_battery_soc(solar_power, load_power, current_soc, dt_hours=1/12):
    """Update battery state of charge"""
    battery_capacity_kwh = 50  # 50 kWh battery
    net_power = solar_power - load_power  # Watts
    
    # Convert to energy (Wh)
    energy_delta = net_power * dt_hours
    
    # Update SOC (with 90% charging/discharging efficiency)
    efficiency = 0.9 if net_power > 0 else 1.0
    soc_delta = (energy_delta * efficiency / (battery_capacity_kwh * 1000)) * 100
    
    new_soc = np.clip(current_soc + soc_delta, 5, 100)  # Never fully discharge
    return new_soc

# Generate dataset
print("Generating microgrid training data...")

data = []
start_date = datetime.now() - timedelta(days=NUM_DAYS)
current_battery_soc = 75.0  # Start at 75%

# Initialize weather variables
cloud_cover = 0.2
weather_condition = "Sunny"

for i in range(TOTAL_SAMPLES):
    timestamp = start_date + timedelta(minutes=5*i)
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    
    # Weather simulation (changes daily)
    if i % (24 * SAMPLES_PER_HOUR) == 0:  # New day
        cloud_cover = np.random.uniform(0, 0.8)
        weather_condition = "Sunny" if cloud_cover < 0.3 else "Partly Cloudy" if cloud_cover < 0.6 else "Cloudy"
    
    # Solar production
    solar_power = generate_solar_production(hour, cloud_cover)
    solar_voltage = 48.0 + np.random.uniform(-1, 1) if solar_power > 0 else 0
    solar_current = solar_power / solar_voltage if solar_voltage > 0 else 0
    
    # Generate readings for each load tier
    total_load_power = 0
    load_data = []
    
    for tier in [1, 2, 3]:
        # Simulate occasional anomalies (5% chance)
        is_anomaly = np.random.random() < 0.05
        
        # Load state (critical loads always on, others vary)
        if tier == 1:
            load_state = 1  # Always on
        else:
            # Loads turn on/off based on time and battery
            load_state = 1 if np.random.random() < 0.7 else 0
        
        if load_state:
            # INA219 voltage reading (12V system with variation)
            voltage = 12.0 + np.random.uniform(-0.5, 0.5)
            
            # ACS712 current reading
            current = generate_load_current(tier, hour, day_of_week, is_anomaly)
            
            # Power calculation
            power = voltage * current
            total_load_power += power
            
            # ACS712 sensor output
            acs712_voltage = current_to_acs712_voltage(current)
            acs712_adc = current_to_adc(acs712_voltage)
            
            # Classification
            classification = "ANOMALY" if is_anomaly else "NORMAL"
            
        else:
            voltage = 0
            current = 0
            power = 0
            acs712_voltage = 1.65  # Zero current
            acs712_adc = current_to_adc(acs712_voltage)
            classification = "NORMAL"
        
        load_data.append({
            'tier': tier,
            'state': load_state,
            'voltage': voltage,
            'current': current,
            'power': power,
            'acs712_voltage': acs712_voltage,
            'acs712_adc': acs712_adc,
            'classification': classification
        })
    
    # Update battery
    current_battery_soc = generate_battery_soc(
        solar_power, 
        total_load_power, 
        current_battery_soc
    )
    
    # Create record for each load
    for load in load_data:
        data.append({
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'hour': hour,
            'day_of_week': day_of_week,
            'tier': load['tier'],
            'load_state': load['state'],
            'ina219_voltage': round(load['voltage'], 2),
            'acs712_current': round(load['current'], 3),
            'acs712_voltage': round(load['acs712_voltage'], 3),
            'acs712_adc': load['acs712_adc'],
            'power_watts': round(load['power'], 2),
            'solar_production': round(solar_power, 2),
            'solar_voltage': round(solar_voltage, 2),
            'solar_current': round(solar_current, 2),
            'battery_soc': round(current_battery_soc, 1),
            'total_load_power': round(total_load_power, 2),
            'cloud_cover': round(cloud_cover * 100, 1),
            'weather': weather_condition,
            'classification': load['classification']
        })
    
    if (i + 1) % 500 == 0:
        print(f"Generated {i + 1}/{TOTAL_SAMPLES} samples...")

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_file = 'microgrid_sensor_data.csv'
df.to_csv(output_file, index=False)

print(f"\n✅ Dataset generated successfully!")
print(f"📁 Saved to: {output_file}")
print(f"\n📊 Dataset Statistics:")
print(f"  Total records: {len(df)}")
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Anomalies: {len(df[df['classification'] == 'ANOMALY'])} ({len(df[df['classification'] == 'ANOMALY'])/len(df)*100:.2f}%)")
print(f"\n🔌 Load breakdown:")
for tier in [1, 2, 3]:
    tier_data = df[df['tier'] == tier]
    print(f"  Tier {tier}: {len(tier_data)} records, avg power: {tier_data['power_watts'].mean():.2f}W")

print(f"\n☀️ Solar production:")
print(f"  Average: {df['solar_production'].mean():.2f}W")
print(f"  Peak: {df['solar_production'].max():.2f}W")

print(f"\n🔋 Battery SOC range: {df['battery_soc'].min():.1f}% - {df['battery_soc'].max():.1f}%")

# Show sample data
print(f"\n📋 Sample data (first 5 rows):")
print(df.head().to_string())
