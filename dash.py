import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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
    if np.random.rand() < 0.1:
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
# Prepare dashboard
# ----------------------
total_packets = []
malicious_cumsum = []
recent_status = []

fig, ax = plt.subplots(figsize=(10,6))
plt.title("AI: Live Offline Simulation")

def update(frame):
    packet = generate_packet()
    X_scaled = scaler.transform([packet])
    prediction = clf.predict(X_scaled)
    threat = 0 if prediction[0]==1 else 1
    status = "MALICIOUS" if threat==1 else "Normal"
    
    # Update lists
    total_packets.append(frame+1)
    malicious_cumsum.append(malicious_cumsum[-1]+1 if threat==1 and malicious_cumsum else 1 if threat==1 else 0 if not malicious_cumsum else malicious_cumsum[-1])
    recent_status.append(status)
    
    # Log to CSV
    df_log = pd.DataFrame([[time.asctime(), packet, status]], columns=['timestamp','features','status'])
    df_log.to_csv("alerts_dashboard.csv", mode='a', index=False, header=False)
    
    # Clear and plot
    ax.clear()
    ax.plot(total_packets, malicious_cumsum, marker='o', color='red', label='Cumulative Malicious')
    ax.set_xlabel("Total Packets Processed")
    ax.set_ylabel("Cumulative Malicious Packets")
    ax.set_title("AI: Live Offline Simulation")
    ax.legend()
    
    # Display last 5 packet statuses on plot
    recent_text = "\n".join([f"{i+1}: {s}" for i,s in zip(range(max(0,len(recent_status)-5), len(recent_status)), recent_status[-5:])])
    ax.text(0.95, 0.95, recent_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(facecolor='white', alpha=0.5))
    
    # Print to console
    print(f"{time.asctime()} | Packet {frame+1}: {packet} -> {status}")

# ----------------------
# Run animation
# ----------------------
ani = FuncAnimation(fig, update, frames=50, interval=500)  # 50 packets, 0.5 sec interval
plt.show()