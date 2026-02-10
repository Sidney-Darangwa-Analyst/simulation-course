# Verify all course packages are installed correctly
import numpy
print('numpy:', numpy.__version__)
import pandas
print('pandas:', pandas.__version__)
import matplotlib
print('matplotlib:', matplotlib.__version__)
import simpy
print('simpy:', simpy.__version__)
import gymnasium
print('gymnasium:', gymnasium.__version__)
import stable_baselines3
print('sb3:', stable_baselines3.__version__)
print("\nAll packages installed successfully!")