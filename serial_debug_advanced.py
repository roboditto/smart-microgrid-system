"""
Advanced serial port scanner - tries all available ports and baud rates.
"""
import serial
import time
import sys

def scan_ports():
    """Find all available COM ports."""
    import platform
    if platform.system() == 'Windows':
        import winreg
        ports = []
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DEVICEMAP\SERIALCOMM')
            i = 0
            while True:
                try:
                    name, value, _ = winreg.EnumValue(key, i)
                    ports.append(value)
                    i += 1
                except OSError:
                    break
        except:
            pass
        return ports if ports else [f'COM{i}' for i in range(1, 10)]
    else:
        import glob
        return glob.glob('/dev/tty*') + glob.glob('/dev/cu*')

def test_port(port, baud=115200, timeout=2):
    """Try to connect and read from a port."""
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2)  # Arduino resets on connection
        
        # Try to read a line
        line = ser.readline()
        if line:
            decoded = line.decode(errors='ignore').strip()
            ser.close()
            return True, decoded
        
        ser.close()
        return False, "No data"
    except Exception as e:
        return False, str(e)

# Scan all ports
print("🔍 Scanning for Arduino...")
print("=" * 80)

ports = scan_ports()
print(f"Found {len(ports)} COM port(s): {ports}\n")

found = False
for port in ports:
    print(f"Testing {port}...", end=" ")
    success, msg = test_port(port, baud=115200)
    
    if success:
        print(f"✅ Found Arduino!")
        print(f"   Port: {port}")
        print(f"   Data: {msg}")
        
        # Read a few more lines
        print(f"\n   Reading more data from {port}:")
        try:
            ser = serial.Serial(port, 115200, timeout=2)
            time.sleep(2)
            for i in range(10):
                line = ser.readline()
                if line:
                    decoded = line.decode(errors='ignore').strip()
                    print(f"   [{i+1}] {decoded}")
            ser.close()
        except:
            pass
        
        found = True
        break
    else:
        print(f"❌ {msg}")

if not found:
    print("\n❌ No Arduino found on any COM port.")
    print("   Try:")
    print("   - Unplug and replug the USB cable")
    print("   - Check Device Manager for the correct COM port")
    print("   - Upload the sketch to the Arduino first (via Arduino IDE or PlatformIO)")
    print("   - Try a different USB cable")

print("\n" + "=" * 80)
print("If found, update COM port in dashboard.py and smartgrid.py accordingly.")
