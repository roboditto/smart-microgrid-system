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
# 1. Initialize Serial
# --------------------------
def connect():
    global ser
    try:
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # Arduino resets on connection
        print("Connected to Arduino on", PORT)
    except Exception as e:
        print("ERROR: Could not connect to Arduino:", e)
        ser = None


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
        line = ser.readline().decode().strip()
        return line
    except Exception:
        connect()
        return ""


def read_packet():
    """Parse CSV from Arduino into values dictionary."""
    line = read_line()
    parts = line.split(",")

    if len(parts) != 7:
        return None  # Bad packet

    try:
        return {
            "millis":         int(parts[0]),
            "voltage":        float(parts[1]),
            "solar_current":  float(parts[2]) / 1000.0,  # mA -> A
            "solar_power":    float(parts[3]) / 1000.0,  # mW -> W
            "load1_current":  float(parts[4]),
            "load2_current":  float(parts[5]),
            "relay_state":    int(parts[6])
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


# --------------------------
# 4. Relay Command
# --------------------------

def set_relay(state: int):
    """Send SETRELAY:0 or SETRELAY:1 to Arduino."""
    global ser
    if ser is None:
        connect()

    state = 1 if state else 0  # ensure valid
    try:
        ser.write(f"SETRELAY:{state}\n".encode())
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
