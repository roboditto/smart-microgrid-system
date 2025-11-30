# Smart Microgrid System

An AI-powered smart microgrid system with renewable energy integration, battery management, predictive analytics, and intelligent load management.

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

## Hardware Wiring & Calibration

- **INA219 (Voltage/Current Sensor)**:
  - `VCC` -> Arduino/RPi `5V`
  - `GND` -> Common ground
  - `SDA` -> Arduino `A4` (SDA)
  - `SCL` -> Arduino `A5` (SCL)
  - `VIN+` -> Positive side of the solar panel / power supply input
  - `VIN-` -> Negative side of the solar panel / power supply input (or return)
  - Note: The INA219 must be placed in series with the supply input to measure solar/battery power. If `VIN+`/`VIN-` are left disconnected the dashboard will fall back to a fixed-voltage assumption for simulated power (12V fallback).

- **ACS712 (Current Sensors, 5A module recommended)**:
  - Wire the load through the ACS712 sensing terminals (not the VCC/GND pins). The current path must pass through the sensor in the correct direction.
  - If measured load is negative, the sensor is oriented backwards — swap the two wires in the sensing path or take the absolute value in software (the dashboard temporarily uses `abs()` for display).
  - Calibration constants (per this repository's hardware testing):
    - `ACS_ZERO` = 2.80 V (measured midpoint voltage on this setup)
    - `ACS_SENSITIVITY` = 0.185 V/A (5A module)

- **Relays (Arduino Uno pins used in firmware)**:
  - `RELAY1_PIN` -> Arduino digital pin `2` (IN1)
  - `RELAY2_PIN` -> Arduino digital pin `4` (IN2)
  - Note: Older README notes listed different GPIO pins (Raspberry Pi mapping) — the Arduino sketch currently uses pins `2` and `4` for relays.

## Serial & Dashboard (Running)

- Default serial port (in `smartgrid.py`): `COM8` on Windows. Update `PORT` or pass an alternate port when connecting.

- Recommended workflow (in project root):

```powershell
# activate virtualenv on Windows
venv\Scripts\activate
# install dependencies
pip install -r requirements.txt
# run Streamlit dashboard
streamlit run dashboard.py
```

- If `streamlit` is not on your PATH but you use the project venv, run:

```powershell
# Windows (use the venv python to run streamlit)
& "<path-to-venv>\Scripts\python.exe" -m streamlit run dashboard.py
```

- To collect real hardware training data: turn **off** Simulation Mode in the dashboard, ensure the Arduino is connected (select the proper COM port), then the app will append rows to `hardware_sensor_data.csv` automatically when readings are available.

## CSV / Data Formats

- **Hardware CSV**: `hardware_sensor_data.csv` — header (current):

```
timestamp,ina219_voltage,ina219_current,ina219_power,acs712_1_current,acs712_2_current,load1_power,load2_power,relay1_state,relay2_state,battery_soc,total_load_power,grid_connected,hour
```

- **Legacy (Simulated) CSV**: `microgrid_sensor_data.csv` — older column names such as `acs712_current`, `power_watts`, and `solar_production` may appear. The dashboard detects format and adapts to both naming schemes.

## Troubleshooting

- INA219 reads zero / dashboard shows 0 W:
  - Ensure `VIN+` and `VIN-` are connected in series with the solar/power input.
  - Verify `SDA`/`SCL` wiring (A4/A5 on Arduino Uno) and common ground.
  - Use `serial_debug.py` or `smartgrid.py` diagnostic functions to view raw packets.

- ACS712 shows negative current or negative power:
  - Swap the two wires through the ACS712 sensing terminal to reverse orientation.
  - If swapping isn't possible immediately, the dashboard displays absolute current values for convenience.

- Serial port not found / `streamlit` not recognized:
  - Activate the same Python environment used to install dependencies.
  - Use the full venv Python to run Streamlit if `streamlit` isn't on PATH (example above).

## Tips & Next Steps

- To generate a larger dataset quickly for model testing, use `generate_training_data.py` which produces simulated `microgrid_sensor_data.csv`.
- Connect INA219 to real solar panel or DC supply for accurate power readings — this improves training on real-world behavior.
- Once you have stable hardware data, disable the temporary 12V fallback and absolute-value workarounds in `dashboard.py` so the system uses true measured values.

---

If you'd like, I can also:
- Add a small wiring diagram image and include it in the repo,
- Update `requirements.txt` to include any missing packages detected while running Streamlit,
- Or open a PR with the README changes.


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
