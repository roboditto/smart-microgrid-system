# Smart Microgrid System

An AI-powered smart microgrid control system with renewable energy integration, battery management, predictive analytics, and intelligent load management.

## Features

### (a) Renewable Energy Input

- **Solar Production Monitoring**: Real-time tracking of solar panel output
- **Weather Integration**: Cloud cover and weather condition simulation
- **Dynamic Power Generation**: Hour-based solar production curves with environmental factors

### (b) Smart AI Energy Controller

- **Anomaly Detection**: IsolationForest ML model trained on 6,000+ historical samples
- **Outage Prediction**: RandomForest regression for 6-hour battery SOC forecasting
- **Real-time Monitoring**: Weather, solar production, battery capacity, and electricity demand

### (c) Load Management Layer

- **Tiered Priority System**:
  - **Tier 1 (Critical)**: Medical equipment, refrigeration, communications - always on
  - **Tier 2 (Essential)**: Lighting, water pumps - high priority
  - **Tier 3 (Non-Critical)**: Other appliances - lowest priority
- **Intelligent Load Shedding**: AI-based automatic load management during power shortages

### (d) Island Mode Operation

- **Automatic Grid Disconnect Detection**: Seamlessly transitions to battery/solar power
- **Priority Load Preservation**: Ensures critical loads remain powered during outages
- **Battery State of Charge Management**: Intelligent charging/discharging based on available power

## Hardware Components

### Sensors

- **INA219**: Voltage and power measurement (12V system, I2C interface)
- **ACS712** (5A version): Current sensing with 185mV/A sensitivity
- **MCP3008**: 10-bit ADC for analog sensor readings (SPI interface)

### Control

- **Raspberry Pi**: Main controller (Python 3.x)
- **Arduino Uno r3**: Sensor data collection
- **3x Relays**: GPIO-controlled load switching (pins 17, 27, 22)

## System Architecture

```text
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                      │
│  (Real-time Monitoring + AI Predictions + Control Panel)    │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ┌───▼────┐         ┌────▼────┐        ┌────▼─────┐
    │ Solar  │         │ Battery │        │  Loads   │
    │ Panels │         │  Bank   │        │ (Tiered) │
    └────────┘         └─────────┘        └──────────┘
        │                   │                   │
    ┌───▼──────────────────▼───────────────────▼───┐
    │         Raspberry Pi Controller              │
    │  (INA219, ACS712, MCP3008, GPIO Relays)      │
    └──────────────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Raspberry Pi (for hardware deployment) or Windows/Linux (for simulation)

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/roboditto/smart-microgrid-system.git
cd smart-microgrid-system
```

2. **Create virtual environment** (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install streamlit pandas plotly scikit-learn numpy
```

4. **For Raspberry Pi hardware (production)**

```bash
pip install RPi.GPIO spidev adafruit-circuitpython-ina219
```

5. **For Windows development (simulation)**
The repository includes mock modules (`spidev.py`, `RPi/GPIO.py`, `board.py`, `busio.py`, `adafruit_ina219.py`) that simulate hardware for testing on non-Raspberry Pi systems.

```bash
pip install -r requirements.txt
```

## Project Structure

```text
smart-microgrid-system/
├── dashboard.py                    # Main Streamlit dashboard (1100+ lines)
├── smartgrid.py                   # Flask-based hardware interface
├── generate_training_data.py      # Training dataset generator
├── microgrid_sensor_data.csv      # Historical sensor data (6,048 records)
├── alerts_scoreboard.csv          # Legacy anomaly log
├── spidev.py                      # Mock SPI module (Windows dev)
├── RPi/GPIO.py                    # Mock GPIO module (Windows dev)
├── board.py                       # Mock board module (Windows dev)
├── busio.py                       # Mock I2C module (Windows dev)
├── adafruit_ina219.py            # Mock INA219 module (Windows dev)
└── README.md                      # This file
```

## Usage

### Generate Training Data

```bash
python generate_training_data.py
```

Creates `microgrid_sensor_data.csv` with 7 days of realistic sensor readings:

- 6,048 records (3 tiers × 12 samples/hour × 24 hours × 7 days)
- Realistic ACS712 and INA219 values
- Solar production curves
- Battery SOC simulation
- ~3-5% anomaly injection

### Run Dashboard (Simulation Mode)

```bash
streamlit run dashboard.py
```

Access at: `http://localhost:8501`

**Dashboard Controls:**

- **Simulation Mode**: Toggle to use simulated sensors (for Windows/testing)
- **Grid Status**: Enable/disable island mode
- **Battery Settings**: Adjust capacity (10-100 kWh) and minimum SOC
- **Power Threshold**: Set maximum load before shedding (10-100W)
- **Auto-refresh**: Enable real-time graph updates (1-10 second intervals)
- **Manual Load Control**: Toggle individual loads by tier

### Run Hardware Interface (Raspberry Pi)

```bash
python smartgrid.py
```

Flask server at: `http://<raspberry-pi-ip>:5000`

## Machine Learning Models

### Anomaly Detection

- **Algorithm**: Isolation Forest
- **Training Data**: 6,048 historical samples
- **Features**: Voltage, current, power, load state, solar production, battery SOC, hour
- **Output**: Real-time anomaly classification (Normal/Anomaly)

### Outage Prediction

- **Algorithm**: Random Forest Regressor
- **Prediction Window**: 6 hours ahead
- **Features**: Hour, solar production, total load power, battery SOC, cloud cover
- **Output**: Battery SOC forecast + outage risk level (Low/Medium/High)

## Monitoring Capabilities

### Real-time Metrics

- Solar production (W, V, A)
- Battery SOC (%), charge/discharge rate
- Load demand per tier (W, V, A)
- Active load count
- Weather conditions (cloud cover, temperature, wind)

### Historical Analysis

- Classification distribution (Normal vs Anomaly)
- Power consumption by tier
- Hourly load patterns
- Solar production curves
- Battery SOC trends
- Sensor statistics (INA219 voltage, ACS712 current, ADC readings)

### AI Predictions

- Anomaly score timeline
- 6-hour battery forecast
- Outage risk assessment
- Load shedding recommendations

## Configuration

### Battery Settings

- **Capacity**: Adjustable 10-100 kWh (default: 50 kWh)
- **Minimum SOC**: Reserve capacity 10-50% (default: 20%)
- **Efficiency**: 90% charging, 100% discharging

### Load Tiers

Edit `LOAD_TIERS` dictionary in `dashboard.py`:

```python
LOAD_TIERS = {
    1: {'name': 'Critical', 'loads': [0], 'priority': 1},
    2: {'name': 'Essential', 'loads': [1], 'priority': 2},
    3: {'name': 'Non-Critical', 'loads': [2], 'priority': 3}
}
```

### Sensor Calibration

For ACS712 (5A version) in `smartgrid.py`:

```python
# 3.3V ADC reference
zero_voltage = 1.65  # Zero current point
sensitivity = 0.122   # 122mV/A (scaled from 185mV/A @ 5V)
```

## Troubleshooting

### Import Errors on Windows

Mock modules are included for development. Ensure they're in the project directory:

- `spidev.py`
- `RPi/GPIO.py`
- `board.py`, `busio.py`, `adafruit_ina219.py`

### Static Graphs

Enable **Auto-refresh** in the sidebar and set refresh interval (1-10 seconds).

### ML Model Not Training

Ensure `microgrid_sensor_data.csv` exists:

```bash
python generate_training_data.py
```

### Type Conversion Errors

Update pandas/numpy/scikit-learn to latest versions:

```bash
pip install --upgrade pandas numpy scikit-learn
```

## Dataset Specifications

### Sensor Data Format

```csv
timestamp,hour,day_of_week,tier,load_state,ina219_voltage,acs712_current,
acs712_voltage,acs712_adc,power_watts,solar_production,solar_voltage,
solar_current,battery_soc,total_load_power,cloud_cover,weather,classification
```

### Sensor Value Ranges

- **INA219 Voltage**: 11.5-12.5V (12V system)
- **ACS712 Current**: 0-5A
- **ACS712 ADC**: 512-1023 (0-1023 for ±5A range)
- **Power**: 0-60W per load
- **Solar Production**: 0-1000W
- **Battery SOC**: 5-100%

## Hardware Wiring (Raspberry Pi)

### GPIO Connections

- Pin 17 (BCM): Relay 1 (Tier 1 - Critical)
- Pin 27 (BCM): Relay 2 (Tier 2 - Essential)
- Pin 22 (BCM): Relay 3 (Tier 3 - Non-Critical)

### SPI (MCP3008 ADC)

- CE0: Chip Select
- MOSI/MISO/SCLK: SPI bus
- Ch0-2: ACS712 outputs for 3 loads

### I2C (INA219)

- SDA: GPIO 2 (Pin 3)
- SCL: GPIO 3 (Pin 5)
- Address: 0x40 (default)

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

- **Repository**: [roboditto/smart-microgrid-system](https://github.com/roboditto/smart-microgrid-system)
- **Issues**: [GitHub Issues](https://github.com/roboditto/smart-microgrid-system/issues)

## Acknowledgments

- Streamlit for the dashboard framework
- Scikit-learn for ML algorithms
- Adafruit for INA219 library
- Raspberry Pi Foundation

---

**Built for**: School Olympiad Coding Competition - Paladin  
**Version**: 2.0  
**Last Updated**: November 2025
