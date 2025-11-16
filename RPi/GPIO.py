"""Mock RPi.GPIO module for Windows development"""

# GPIO modes
BCM = 11
BOARD = 10

# GPIO directions
IN = 1
OUT = 0

# GPIO states
HIGH = 1
LOW = 0

def setmode(mode):
    """Set GPIO numbering mode"""
    print(f"[Mock GPIO] setmode({mode})")

def setup(pin, direction):
    """Setup GPIO pin"""
    print(f"[Mock GPIO] setup(pin={pin}, direction={direction})")

def output(pin, state):
    """Set GPIO output state"""
    print(f"[Mock GPIO] output(pin={pin}, state={state})")

def input(pin):
    """Read GPIO input state"""
    print(f"[Mock GPIO] input(pin={pin})")
    return LOW

def cleanup():
    """Cleanup GPIO settings"""
    print("[Mock GPIO] cleanup()")
