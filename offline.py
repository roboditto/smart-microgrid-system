import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

# ----------------------
# Initialize AI model with simulated data
# ----------------------
def simulate_initial_data(n_samples=500):
    data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.1:  # 10% malicious
            packet = [np.random.normal(1000, 100),
                      np.random.normal(200, 50),
                      np.random.choice([0,1]),
                      np.random.normal(50, 20)]
        else:
            packet = [np.random.normal(500, 50),
                      np.random.normal(50, 10),
                      np.random.choice([0,1]),
                      np.random.normal(10, 5)]
        data.append(packet)
    return pd.DataFrame(data, columns=['packet_size','duration','protocol_type','num_connections'])

df_train = simulate_initial_data()
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(df_train)
clf = IsolationForest(contamination=0.1, random_state=42)
clf.fit(X_train_scaled)
print("Model trained with simulated initial data.")

# ----------------------
# Simulate streaming packets
# ----------------------
def generate_packet():
    if np.random.rand() < 0.1:  # 10% malicious
        return [np.random.normal(1000, 100),
                np.random.normal(200, 50),
                np.random.choice([0,1]),
                np.random.normal(50, 20)]
    else:
        return [np.random.normal(500, 50),
                np.random.normal(50, 10),
                np.random.choice([0,1]),
                np.random.normal(10, 5)]

# ----------------------
# Real-time detection, logging, and alerting
# ----------------------
malicious_counts = []
timestamps = []

for i in range(50):  # simulate 50 packets
    packet = generate_packet()
    X_scaled = scaler.transform([packet])
    prediction = clf.predict(X_scaled)
    threat = 0 if prediction[0]==1 else 1
    status = "MALICIOUS" if threat==1 else "Normal"
    
    # Print to console
    print(f"{time.asctime()} | Packet {i+1}: {packet} -> {status}")
    
    # Log to CSV
    df_log = pd.DataFrame([[time.asctime(), packet, status]], columns=['timestamp','features','status'])
    df_log.to_csv("alerts_sim.csv", mode='a', index=False, header=False)
    
    # Count for visualization
    if threat == 1:
        malicious_counts.append(1)
    else:
        malicious_counts.append(0)
    timestamps.append(i)
    
    # Optional: simulate alert
    if threat == 1:
        print(">>> ALERT: Malicious packet detected!")

    time.sleep(0.3)  # simulate real-time arrival

# ----------------------
# Simple visualization
# ----------------------
plt.figure(figsize=(10,5))
plt.plot(timestamps, np.cumsum(malicious_counts), marker='o')
plt.xlabel("Packet Number")
plt.ylabel("Cumulative Malicious Packets")
plt.title("Offline Threat Detection Simulation")
plt.grid(True)
plt.show()
