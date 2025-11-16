import RPi.GPIO as GPIO
import time
import spidev
import board
import busio
from adafruit_ina219 import INA219
from flask import Flask, render_template
import plotly.graph_objects as go

# GPIO Setup
GPIO.setmode(GPIO.BCM)
relay_pins = [17, 27, 22]  # Relays for loads
for pin in relay_pins:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

# MCP3008 Setup (for ACS712)
spi = spidev.SpiDev()
spi.open(0, 0)
spi.max_speed_hz = 1350000

def read_adc(channel):
    """Read analog value from MCP3008"""
    adc = spi.xfer2([1, (8+channel)<<4, 0])
    data = ((adc[1]&3) << 8) + adc[2]
    return data

def read_current(channel):
    """Convert ADC value to Amps (ACS712 5A version)"""
    adc_val = read_adc(channel)
    voltage = (adc_val * 3.3) / 1023  # ADC to voltage
    current = (voltage - 2.5) / 0.185  # ACS712 formula
    return max(current, 0)

# INA219 Setup (for voltage)
SHUNT_OHMS = 0.1
ina = INA219(SHUNT_OHMS)
ina.configure()

def read_power(channel):
    """Compute power for a load in mW: voltage*current"""
    current = read_current(channel)
    voltage = ina.voltage()
    return voltage * current

#  Threshold
THRESHOLD = 25  # Watts

# Flask Dashboard
app = Flask(__name__)

@app.route('/')
def dashboard():
    powers = [read_power(i) for i in range(len(relay_pins))]
    total_power = sum(powers)

    # Load management
    if total_power > THRESHOLD:
        for i, pin in enumerate(relay_pins):
            if powers[i] < 15:  # non-critical loads
                GPIO.output(pin, GPIO.LOW)
    else:
        for pin in relay_pins:
            GPIO.output(pin, GPIO.HIGH)

    # Plot graph
    fig = go.Figure()
    fig.add_bar(x=[f'Load {i+1}' for i in range(len(powers))], y=powers)
    fig.update_layout(title=f'Total Power: {total_power:.2f} W')
    graph_html = fig.to_html(full_html=False)
    return render_template("dashboard.html", graph_html=graph_html)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)

# pip install RPi.GPIO to import RPI libraries