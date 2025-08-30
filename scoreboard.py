import pandas as pd
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import yagmail

# ----------------------
# Email alert setup
# ----------------------
EMAIL_USER = "yourdemo@gmail.com"
EMAIL_PASSWORD = "yourpassword"  
EMAIL_RECEIVER = "receiver@gmail.com"

def send_alert(packet_info, severity):
    try:
        yag = yagmail.SMTP(EMAIL_USER, EMAIL_PASSWORD)
        yag.send(EMAIL_RECEIVER, "Threat Detected!", 
                 f"Threat detected: {packet_info}\nSeverity Score: {severity}")
        print(">>> ALERT: Email sent!")
    except Exception as e:
        print(f"Email failed: {e}")

# ----------------------
# Initialize AI model
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
# Packet generator
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
# Compute severity
# ----------------------
def compute_severity(packet):
    score = 0
    score += max(0, packet[0]-600)/400
    score += max(0, packet[1]-60)/200
    score += packet[2]*0.5
    score += max(0, packet[3]-20)/50
    return round(score,2)

# ----------------------
# Dashboard setup
# ----------------------
total_packets = []
malicious_cumsum = []
severity_scores = []

fig, ax = plt.subplots(figsize=(10,6))
plt.title("AI: Offline Simulation with Scoreboard")

def update(frame):
    packet = generate_packet()
    X_scaled = scaler.transform([packet])
    prediction = clf.predict(X_scaled)
    threat = 0 if prediction[0]==1 else 1
    status = "MALICIOUS" if threat==1 else "Normal"
    
    severity = compute_severity(packet)
    
    # Update stats
    total_packets.append(frame+1)
    malicious_cumsum.append(malicious_cumsum[-1]+1 if threat==1 and malicious_cumsum else 1 if threat==1 else 0 if not malicious_cumsum else malicious_cumsum[-1])
    severity_scores.append(severity)
    
    # Log CSV
    df_log = pd.DataFrame([[time.asctime(), packet, status, severity]], 
                          columns=['timestamp','features','status','severity'])
    df_log.to_csv("alerts_scoreboard.csv", mode='a', index=False, header=False)
    
    # Email alert
    if threat==1 and severity>=1.0:
        send_alert(packet, severity)
    
    # Clear and plot
    ax.clear()
    ax.plot(total_packets, malicious_cumsum, marker='o', color='red', label='Cumulative Malicious')
    ax.set_xlabel("Total Packets Processed")
    ax.set_ylabel("Cumulative Malicious Packets")
    ax.set_title("AI: Offline Simulation with Scoreboard")
    ax.legend()
    
    # Scoreboard stats
    total = len(total_packets)
    malicious = malicious_cumsum[-1] if malicious_cumsum else 0
    max_sev = max(severity_scores) if severity_scores else 0
    success_rate = (malicious/total)*100 if total>0 else 0
    
    scoreboard_text = (f"Total Packets: {total}\n"
                       f"Malicious Detected: {malicious}\n"
                       f"Max Severity: {max_sev}\n"
                       f"Detection Rate: {success_rate:.2f}%")
    
    ax.text(0.02, 0.95, scoreboard_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    # Print to console
    print(f"{time.asctime()} | Packet {frame+1}: {packet} -> {status}, Severity: {severity}")

# ----------------------
# Run animation
# ----------------------
ani = FuncAnimation(fig, update, frames=50, interval=500)
plt.show()
