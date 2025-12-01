#!/usr/bin/env python3
"""
Initialize the hardware_sensor_data.csv file with headers.
Run this once before starting the dashboard to collect hardware data.
"""
import os

CSV_FILE = 'hardware_sensor_data.csv'
HEADERS = (
    'timestamp,ina219_voltage,ina219_current,ina219_power,'
    'acs712_1_current,acs712_2_current,load1_power,load2_power,'
    'relay1_state,relay2_state,battery_soc,total_load_power,'
    'grid_connected,hour\n'
)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, 'w') as f:
        f.write(HEADERS)
    print(f"Created {CSV_FILE} with headers")
else:
    print(f"ℹ️  {CSV_FILE} already exists")
    print(f"   Current size: {os.path.getsize(CSV_FILE)} bytes")
