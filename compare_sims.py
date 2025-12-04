import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compare_runs(csv_file1, csv_file2):
    # Load both CSV files
    data1 = pd.read_csv(csv_file1)
    data2 = pd.read_csv(csv_file2)

    # --- Position over time ---
    plt.figure()
    plt.plot(data1['time'], data1['px'], label='X Position Run 1', color='blue')
    plt.plot(data1['time'], data1['py'], label='Y Position Run 1', color='cyan')
    plt.plot(data1['time'], data1['pz'], label='Z Position Run 1', color='navy')

    plt.plot(data2['time'], data2['px'], label='X Position Run 2', linestyle='--', color='red')
    plt.plot(data2['time'], data2['py'], label='Y Position Run 2', linestyle='--', color='orange')
    plt.plot(data2['time'], data2['pz'], label='Z Position Run 2', linestyle='--', color='maroon')

    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.title('Car Position Over Time')
    plt.legend()
    plt.show()

    # --- Linear speed over time ---
    vel1 = np.sqrt(data1['vx']**2 + data1['vy']**2 + data1['vz']**2)
    vel2 = np.sqrt(data2['vx']**2 + data2['vy']**2 + data2['vz']**2)

    plt.figure()
    plt.plot(data1['time'], vel1, label='Speed Run 1', color='blue')
    plt.plot(data2['time'], vel2, label='Speed Run 2', linestyle='--', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.title('Car Linear Speed Over Time')
    plt.legend()
    plt.show()

    # --- Collision events ---
    plt.figure()
    plt.plot(data1['time'], data1['collision'], label='Collision Run 1', color='blue', drawstyle='steps-post')
    plt.plot(data2['time'], data2['collision'], label='Collision Run 2', color='red', linestyle='--', drawstyle='steps-post')
    plt.xlabel('Time (s)')
    plt.ylabel('Collision (0/1)')
    plt.title('Collision Events')
    plt.legend()
    plt.show()
