from scapy.all import sniff
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import threading


# Initialize model

# For initial training, simulate network traffic
def simulate_initial_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.1:  # 10% malicious
            packet = [
                np.random.normal(1000, 100),  # packet_size
                np.random.normal(200, 50),    # duration
                np.random.choice([0,1]),      # protocol_type
                np.random.normal(50, 20)      # num_connections
            ]
        else:
            packet = [
                np.random.normal(500, 50),
                np.random.normal(50, 10),
                np.random.choice([0,1]),
                np.random.normal(10, 5)
            ]
        data.append(packet)
    return pd.DataFrame(data, columns=['packet_size','duration','protocol_type','num_connections'])

df_train = simulate_initial_data()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train_scaled)
print("Model trained with initial simulated data.")


# Packet feature extractor

def extract_features(packet):
    """
    Extract simplified features from a packet
    Features: packet size, protocol type, duration (simulated), num_connections (simulated)
    """
    try:
        packet_size = len(packet)
        protocol_type = 0 if packet.haslayer('TCP') else 1  # 0=TCP,1=UDP
        duration = np.random.normal(50, 10)  # Simulated for demo
        num_connections = np.random.normal(10, 5)  # Simulated for demo
        return [packet_size, duration, protocol_type, num_connections]
    except:
        return None


# Step 3: Real-time detection & logging

def process_packet(packet):
    features = extract_features(packet)
    if features:
        X_scaled = scaler.transform([features])
        prediction = clf.predict(X_scaled)
        threat = 0 if prediction[0]==1 else 1
        status = "MALICIOUS" if threat==1 else "Normal"
        print(f"Packet: {features} -> {status}")
        # Log detected threats
        with open("alerts.log","a") as f:
            f.write(f"{time.asctime()} | {features} | {status}\n")


# Start live sniffing (non-blocking)

print("Starting live network monitoring...")
sniffer_thread = threading.Thread(target=lambda: sniff(prn=process_packet, store=False))
sniffer_thread.start()


# Optional Visualization

def live_plot():
    malicious_counts = []
    timestamps = []
    while True:
        try:
            df = pd.read_csv("alerts.log", delimiter="|", header=None, names=["time","features","status"])
            count = len(df[df['status'].str.contains("MALICIOUS")])
            malicious_counts.append(count)
            timestamps.append(time.time())
            plt.clf()
            plt.plot(timestamps, malicious_counts, label="Malicious Packets Detected")
            plt.xlabel("Time")
            plt.ylabel("Count")
            plt.legend()
            plt.pause(1)
        except:
            pass

# Uncomment below to run live plotting
# plt.ion()
# live_plot()
