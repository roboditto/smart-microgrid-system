"""Mock busio module for Windows development"""

class I2C:
    """Mock I2C bus"""
    def __init__(self, scl, sda):
        print(f"[Mock I2C] Initialized with SCL={scl}, SDA={sda}")
    
    def writeto(self, address, buffer):
        print(f"[Mock I2C] Write to {address}: {buffer}")
    
    def readfrom_into(self, address, buffer):
        print(f"[Mock I2C] Read from {address}")
