"""
Mock spidev module for Windows development.
This allows code to run on Windows without errors.
Replace with actual spidev on Raspberry Pi.
"""

class SpiDev:
    def __init__(self):
        self.mode = 0
        self.max_speed_hz = 500000
        self.bits_per_word = 8
        
    def open(self, bus, device):
        """Mock open - does nothing on Windows"""
        print(f"[MOCK] SPI opened: bus={bus}, device={device}")
        
    def close(self):
        """Mock close - does nothing on Windows"""
        print("[MOCK] SPI closed")
        
    def xfer(self, data):
        """Mock transfer - returns dummy data"""
        print(f"[MOCK] SPI transfer: {data}")
        return [0] * len(data)
        
    def xfer2(self, data):
        """Mock transfer2 - returns dummy data"""
        print(f"[MOCK] SPI transfer2: {data}")
        return [0] * len(data)
        
    def readbytes(self, num):
        """Mock read - returns dummy data"""
        print(f"[MOCK] SPI read {num} bytes")
        return [0] * num
        
    def writebytes(self, data):
        """Mock write - does nothing on Windows"""
        print(f"[MOCK] SPI write: {data}")
