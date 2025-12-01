import serial
import time

# --------------------------
# SERIAL PORT CONFIGURATION
# --------------------------
# Adjust if needed:
#   Windows: 'COM3'
#   RPi / Linux: '/dev/ttyACM0' or '/dev/ttyUSB0'
# --------------------------
PORT = 'COM8'
#PORT = "/dev/ttyACM0" uncomment for unix systems eg. RPi
BAUD = 115200

# Global serial object
ser = None

# --------------------------
# SENSOR SCALING & CALIBRATION
# --------------------------
# Adjust these if your sensor readings are off:
ACS712_SCALE = 1.0          # Multiplier for ACS712 current (if raw ADC, may need 0.185 A/LSB or similar)
BATTERY_VOLTAGE_SCALE = 1.0 # Multiplier for battery voltage divider (if using analog input)
BATTERY_VOLTAGE_PIN = None  # Analog pin for battery (Arduino would send as 8th field if available)


# --------------------------
# 1. Initialize Serial
# --------------------------
def connect(port=None, baud=None):
    """Connect to the serial device. Optional `port` and `baud` override module defaults."""
    global ser, PORT, BAUD
    if port:
        PORT = port
    if baud:
        BAUD = baud

    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # Arduino resets on connection
        print("Connected to Arduino on", PORT)
    except Exception as e:
        print("ERROR: Could not connect to Arduino:", e)
        ser = None


def disconnect():
    """Close serial connection if open."""
    global ser
    try:
        if ser is not None and hasattr(ser, 'is_open') and ser.is_open:
            ser.close()
            print("Serial disconnected")
    except Exception:
        pass
    finally:
        ser = None


def is_connected():
    """Return True if serial port is open and ready."""
    return ser is not None and hasattr(ser, 'is_open') and ser.is_open


# Attempt initial connection
connect()


# --------------------------
# 2. Read & Parse Arduino CSV
# Format:
# millis, V, mA, mW, ACS1(A), ACS2(A), relay_state
# --------------------------
def read_line():
    """Reads one line from Arduino, reconnects if needed."""
    global ser
    if ser is None or not ser.is_open:
        connect()

    try:
        raw = ser.readline()
        if not raw:
            return ""

        if isinstance(raw, bytes):
            # Ignore decode errors to avoid exceptions from noisy serial
            line = raw.decode(errors='ignore').strip()
        else:
            line = str(raw).strip()

        return line
    except Exception:
        connect()
        return ""


def read_packet():
    """Parse CSV from Arduino into values dictionary.
    
    Format: millis,voltage,current_mA,power_mW,load1_current_A,load2_current_A,relay1_state,relay2_state
    """
    line = read_line()
    parts = line.split(",")

    if len(parts) < 7:
        return None  # Bad packet

    try:
        return {
            "millis":         int(parts[0]),
            "voltage":        float(parts[1]),
            "solar_current":  float(parts[2]) / 1000.0,  # mA -> A
            "solar_power":    float(parts[3]) / 1000.0,  # mW -> W
            "load1_current":  float(parts[4]) * ACS712_SCALE,
            "load2_current":  float(parts[5]) * ACS712_SCALE,
            "relay_state":    int(parts[6]),  # Relay 1
            "relay2_state":   int(parts[7]) if len(parts) > 7 else 0,  # Relay 2
            "battery_voltage": float(parts[8]) if len(parts) > 8 else None,  # Optional 9th field
        }
    except:
        return None


# --------------------------
# 3. Public Sensor Functions
# --------------------------

def read_voltage():
    pkt = read_packet()
    return pkt["voltage"] if pkt else 0.0


def read_solar_current():
    pkt = read_packet()
    return pkt["solar_current"] if pkt else 0.0


def read_solar_power():
    pkt = read_packet()
    return pkt["solar_power"] if pkt else 0.0


def read_current(channel):
    """channel: 0 or 1 -> ACS712 #1 or ACS712 #2"""
    pkt = read_packet()
    if not pkt:
        return 0.0

    if channel == 0:
        return pkt["load1_current"]
    elif channel == 1:
        return pkt["load2_current"]
    else:
        return 0.0


def read_relay_state():
    pkt = read_packet()
    return pkt["relay_state"] if pkt else 0


def read_battery_voltage():
    """Read battery voltage from 8th field (if Arduino sends it) or return None."""
    pkt = read_packet()
    if pkt and pkt.get("battery_voltage"):
        return pkt["battery_voltage"] * BATTERY_VOLTAGE_SCALE
    return None


def get_diagnostic_info():
    """Return a dictionary of raw sensor readings for debugging."""
    pkt = read_packet()
    if not pkt:
        return {"error": "No packet"}
    
    return {
        "millis": pkt["millis"],
        "ina219_voltage_v": pkt["voltage"],
        "ina219_current_a": pkt["solar_current"],
        "ina219_power_w": pkt["solar_power"],
        "acs712_1_raw_a": pkt["load1_current"],
        "acs712_2_raw_a": pkt["load2_current"],
        "relay_state": pkt["relay_state"],
        "battery_voltage_v": pkt.get("battery_voltage"),
    }


# --------------------------
# 4. Relay Commands
# --------------------------

def set_relay(state: int, relay: int = 1):
    """Send relay command to Arduino.
    
    Args:
        state: 0 or 1 (OFF or ON)
        relay: 1 or 2 (which relay to control, default 1)
    
    Examples:
        set_relay(1)        # Turn relay 1 ON
        set_relay(0, relay=2)  # Turn relay 2 OFF
    """
    global ser
    if ser is None:
        connect()

    state = 1 if state else 0  # ensure valid
    relay = 1 if relay == 1 else 2  # ensure valid relay number
    
    try:
        if relay == 1:
            # Legacy single-relay format (for backward compatibility)
            ser.write(f"SETRELAY:{state}\n".encode())
        else:
            # Read current relay 1 state and send both
            pkt = read_packet()
            relay1 = pkt["relay_state"] if pkt else 0
            ser.write(f"SETRELAY:{relay1},{state}\n".encode())
        
        time.sleep(0.1)
    except:
        connect()


def set_relays(relay1: int, relay2: int):
    """Set both relays at once.
    
    Args:
        relay1: 0 or 1 (OFF or ON)
        relay2: 0 or 1 (OFF or ON)
    
    Example:
        set_relays(1, 0)  # Relay 1 ON, Relay 2 OFF
    """
    global ser
    if ser is None:
        connect()

    relay1 = 1 if relay1 else 0
    relay2 = 1 if relay2 else 0
    
    try:
        ser.write(f"SETRELAY:{relay1},{relay2}\n".encode())
        time.sleep(0.1)
    except:
        connect()


# --------------------------
# Debug Runner
# --------------------------
if __name__ == "__main__":
    while True:
        pkt = read_packet()
        if pkt:
            print(pkt)
        time.sleep(1)
