"""
Quick serial monitor to see raw Arduino output and diagnose sensor issues.
Run this, connect to your Arduino, and let it print 10-20 lines.
"""
import serial
import time

PORT = 'COM8'
BAUD = 115200

try:
    ser = serial.Serial(PORT, BAUD, timeout=2)
    time.sleep(2)
    print(f"✅ Connected to {PORT} @ {BAUD} baud\n")
    print("Raw Arduino Output:")
    print("-" * 80)
    
    for i in range(20):
        line = ser.readline()
        if line:
            # Print both raw bytes and decoded
            print(f"[{i+1}] Raw bytes: {line}")
            try:
                decoded = line.decode(errors='ignore').strip()
                print(f"     Decoded:   {decoded}")
                # Try to parse as CSV
                parts = decoded.split(',')
                print(f"     Fields:    {len(parts)} parts")
                for j, part in enumerate(parts):
                    print(f"       [{j}] = {part}")
            except Exception as e:
                print(f"     Error:     {e}")
            print()
        else:
            print(f"[{i+1}] No data (timeout)")
        time.sleep(0.5)
    
    ser.close()
    print("\nDone. Check the output above for:")
    print("  - Are all 7 fields present? (millis, V, mA, mW, ACS1, ACS2, relay)")
    print("  - Are ACS712 values (fields 4 & 5) zero or very small?")
    print("  - What is the INA219 voltage reading (field 1)?")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print(f"   Check that {PORT} is correct and Arduino is connected.")
