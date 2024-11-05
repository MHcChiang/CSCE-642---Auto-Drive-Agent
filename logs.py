import numpy as np
import matplotlib.pyplot as plt


mdl_name = "CarRacing_test"

# Initialize SAC
log_path = "logs/" + mdl_name + "/evaluations.npz"
# Load the file
data = np.load(log_path)

# List all available keys
print("Keys in the file:", data.files)

time_steps = data['timesteps']
results = data['results']
episode_lengths = data['ep_lengths']
data.close()

results = np.mean(results, axis= 1)
plt.figure(figsize=(8, 6), dpi=150)
plt.plot(time_steps, results)
plt.show()