"""Mock adafruit_ina219 module for Windows development"""

class INA219:
    """Mock INA219 voltage/current sensor"""
    def __init__(self, shunt_ohms, i2c=None, address=0x40):
        self.shunt_ohms = shunt_ohms
        print(f"[Mock INA219] Initialized with shunt={shunt_ohms}Ω")
    
    def configure(self, voltage_range=1, gain=1, bus_adc=1, shunt_adc=1):
        """Configure sensor settings"""
        print(f"[Mock INA219] configure() called")
    
    def voltage(self):
        """Return bus voltage in volts"""
        import random
        voltage = 12.0 + random.uniform(-0.5, 0.5)
        return voltage
    
    def current(self):
        """Return current in milliamps"""
        import random
        current = random.uniform(100, 500)
        return current
    
    def power(self):
        """Return power in milliwatts"""
        return self.voltage() * self.current()
