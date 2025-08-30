from scapy.all import sniff
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import threading
import yagmail

# ----------------------
# Initialize email alerts
# ----------------------
EMAIL_USER = "yourdemo@gmail.com"      # Replace with your demo email
EMAIL_PASSWORD = "yourpassword"        # Replace with app password
EMAIL_RECEIVER = "receiver@gmail.com"  # Where alerts will go

def send_alert(packet_info):
    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
        yag.send(EMAIL_RECEIVER, "Threat Detected!", f"Threat detected: {packet_info}")
        print("Alert sent via email!")
    except Exception as e:
        print(f"Email failed: {e}")

# ----------------------
# Initialize AI model
# ----------------------
def simulate_initial_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.1:
            packet = [np.random.normal(1000, 100), np.random.normal(200, 50), np.random.choice([0,1]), np.random.normal(50, 20)]
        else:
            packet = [np.random.normal(500, 50), np.random.normal(50, 10), np.random.choice([0,1]), np.random.normal(10, 5)]
        data.append(packet)
    return pd.DataFrame(data, columns=['packet_size','duration','protocol_type','num_connections'])

df_train = simulate_initial_data()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train_scaled)
print("Model trained with initial simulated data.")

# ----------------------
# Feature extraction
# ----------------------
def extract_features(packet):
    try:
        packet_size = len(packet)
        protocol_type = 0 if packet.haslayer('TCP') else 1
        duration = np.random.normal(50, 10)
        num_connections = np.random.normal(10, 5)
        return [packet_size, duration, protocol_type, num_connections]
    except:
        return None

# ----------------------
# Real-time detection and logging
# ----------------------
def process_packet(packet):
    features = extract_features(packet)
    if features:
        X_scaled = scaler.transform([features])
        prediction = clf.predict(X_scaled)
        threat = 0 if prediction[0]==1 else 1
        status = "MALICIOUS" if threat==1 else "Normal"
        print(f"{time.asctime()} | Packet: {features} -> {status}")

        # Log to CSV
        df_log = pd.DataFrame([[time.asctime(), features, status]], columns=['timestamp','features','status'])
        df_log.to_csv("alerts.csv", mode='a', index=False, header=False)

        # Send alert if malicious
        if threat == 1:
            send_alert(features)

# ----------------------
# Start live sniffing in a separate thread
# ----------------------
print("Starting live network monitoring...")
sniffer_thread = threading.Thread(target=lambda: sniff(prn=process_packet, store=False))
sniffer_thread.start()

# ----------------------
# Live visualization dashboard
# ----------------------
def live_plot():
    plt.ion()
    malicious_counts = []
    timestamps = []
    while True:
        try:
            df = pd.read_csv("alerts.csv", names=['timestamp','features','status'])
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
