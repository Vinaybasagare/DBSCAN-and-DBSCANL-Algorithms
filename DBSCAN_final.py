import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import ast
import os
import numpy as np
from datetime import timedelta
from scipy.signal import firwin, filtfilt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Specify the folder containing CSV files
df = pd.read_csv("D:\Wireshark\Probe requests\CSVFILES\sc6-61_2024-02-21_ver-99.csv", sep=";", 
                 usecols=['datetime', 'src', 'randomized', 'rssi', 'dot11elt','src_vendor','occupancy'])
df['datetime'] = pd.to_datetime(df['datetime'])
vendors_to_remove = ['Intel Corporate', 'Espressif Inc.', 'Raspberry Pi Trading Ltd', 
                     'TP-LINK TECHNOLOGIES CO.,LTD.','Liteon Technology Corporation','ASUSTek COMPUTER INC.']

df = df[~df['src_vendor'].isin(vendors_to_remove)]

# Filter based on RSSI range
min_value = -70
max_value = -30
mask = (df['rssi'] >= min_value) & (df['rssi'] <= max_value)
df = df[mask]
df_fake = df[df['randomized'] == 1].drop_duplicates(subset=['src'])
df_real = df[df['randomized'] == 0].drop_duplicates(subset=['src'])

# Convert MAC addresses to integers
def mac_to_integer(mac):
    return int(''.join(mac.split(':')), 16) if mac else None

df_real['src'] = df_real['src'].apply(mac_to_integer)
df['src'] = df['src'].apply(mac_to_integer)
df_fake['src'] = df_fake['src'].apply(mac_to_integer)

# Filter fake addresses
df_fake['timestamp'] = df_fake['datetime'].apply(lambda x: x.timestamp())
df_fake['src_scaled'] = StandardScaler().fit_transform(df_fake[['src']])

# Convert string representations of lists in the dot11elt column to actual lists
def convert_to_lists(df):
    df['dot11elt'] = df['dot11elt'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

convert_to_lists(df_fake)
# Define the Jaccard similarity function
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# Define the range_query_jaccard function
def range_query_jaccard(df, target_mac, dr_threshold=1):
    target_data = df.loc[df['src'] == target_mac, 'dot11elt']
    if target_data.empty:
        return []

    target_data_rates = set(target_data.values[0])

    def calculate_jaccard(dr):
        if isinstance(dr, list) and dr:
            return jaccard_similarity(target_data_rates, set(dr))
        else:
            return 0

    df_copy = df.copy()
    df_copy['jaccard_similarity'] = df_copy['dot11elt'].apply(calculate_jaccard)

    final_neighbors = df_copy[df_copy['jaccard_similarity'] >= dr_threshold]
    final_neighbors = final_neighbors[final_neighbors['src'] != target_mac]

    return final_neighbors['src'].tolist()

# Function to perform two-layer DBSCAN clustering
def two_layer_dbscan(df, rssi_eps=0.7, dr_threshold=1, min_samples=4):
    # First layer DBSCAN based on RSSI
    df = df.copy()
    dbscan_rssi = DBSCAN(eps=rssi_eps, min_samples=min_samples)
    #df['cluster'] = dbscan_rssi.fit_predict(df[['rssi']])
    df.loc[:, 'cluster'] = dbscan_rssi.fit_predict(df[['rssi']])
    #print("Clusters after RSSI DBSCAN:")
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_df = df[df['cluster'] == cluster_id]
        #print(f"Cluster ID: {cluster_id}")
        #print(cluster_df[['src', 'rssi', 'dot11elt']])

    final_clusters = []

    # Process each cluster found by RSSI-based DBSCAN
    for cluster_id in df['cluster'].unique():
        if cluster_id == -1:
            continue
        cluster_df = df[df['cluster'] == cluster_id]
        all_neighbors = {}

        # Find neighbors based on Jaccard similarity within the cluster
        for mac in cluster_df['src'].unique():
            neighbors = range_query_jaccard(cluster_df, mac, dr_threshold=dr_threshold)
            all_neighbors[mac] = neighbors

        # Create refined clusters based on the Jaccard similarity results
        for mac, neighbors in all_neighbors.items():
            if len(neighbors) >= 2:  # At least min_samples-1 neighbors to form a cluster
                final_clusters.append([mac] + neighbors)

    # Remove duplicate clusters
    unique_clusters = set(frozenset(cluster) for cluster in final_clusters)
    final_unique_clusters = [list(cluster) for cluster in unique_clusters]

    return final_unique_clusters

start_time = df['datetime'].min()
end_time = df['datetime'].max()
start_time = pd.to_datetime(start_time)
end_time = pd.to_datetime(end_time)
interval_duration = timedelta(minutes=30)
step = timedelta(minutes=1)

intervals = []
current_start = start_time
while current_start + interval_duration <= end_time:
    current_end = current_start + interval_duration
    intervals.append((current_start, current_end))
    current_start += step

# Process each interval
results = []

for start, end in intervals:
    group = df_fake[(df_fake['datetime'] >= start) & (df_fake['datetime'] < end)]

    if not group.empty:
        final_clusters = two_layer_dbscan(group)
        num_final_clusters = len(final_clusters)
        num_real_devices = df_real[(df_real['datetime'] >= start) & (df_real['datetime'] < end)]['src'].nunique()
        total_devices = num_final_clusters + num_real_devices
        max_occupancy = group['occupancy'].max()

        closest_record = group.iloc[-1]
        occupancy_at_end_time = closest_record['occupancy']

        results.append({
            'interval_start': start,
            'interval_end': end,
            #'num_final_clusters': num_final_clusters,
            #'num_real_devices': num_real_devices,
            'total_devices': total_devices,
            'max_occupancy': max_occupancy,
            'occupancy_at_end_time': occupancy_at_end_time
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
filtered_df = results_df[results_df['occupancy_at_end_time'] > 1]  # Example threshold
window_size = 20

# Calculate absolute differences using the filtered dataframe
filtered_df['smoothed_total_devices1'] = filtered_df['total_devices'].rolling(window=window_size).mean()
filtered_df['absolute_difference1'] = abs(filtered_df['smoothed_total_devices1'] - filtered_df['occupancy_at_end_time'])

# Calculate the moving average and MAE
mae1 = filtered_df['absolute_difference1'].mean()

# Print the MAE
#print(f"Mean Absolute Error (MAE1) smoothed: {mae1:.2f}")

# Filter design parameters
filter_order = 20 # Order of the filter
cutoff_frequency = 0.1  # Normalized cutoff frequency (0.1 means 10% of the Nyquist frequency)

# Design the FIR filter using a Hamming window
fir_coefficients = firwin(filter_order, cutoff_frequency, window='hamming')

# Smooth the total_devices data
smoothed_total_devices = filtfilt(fir_coefficients, 1.0, filtered_df['total_devices'])
# Smooth the occupancy_at_end_time data
smoothed_occupancy_at_end_time = filtfilt(fir_coefficients, 1.0, filtered_df['occupancy_at_end_time'])

# Trim points to handle edge effects
trim_points = filter_order // 2

# Trim the smoothed data
trimmed_smoothed_total_devices = smoothed_total_devices[trim_points:-trim_points]
trimmed_smoothed_occupancy_at_end_time = smoothed_occupancy_at_end_time[trim_points:-trim_points]

# Also trim the corresponding time intervals to match the length of the trimmed data
trimmed_time_intervals = filtered_df['interval_end'][trim_points:-trim_points]

# Trim the original data to match the length of the smoothed and trimmed data
trimmed_original_total_devices = filtered_df['total_devices'][trim_points:-trim_points]
trimmed_original_occupancy_at_end_time = filtered_df['occupancy_at_end_time'][trim_points:-trim_points]


# Plot the results
plt.figure(figsize=(12, 6))

# Plot original data
plt.plot(filtered_df['interval_end'], filtered_df['total_devices'], label='Original Total Devices', marker='o', linestyle='-', alpha=0.5)
plt.plot(filtered_df['interval_end'], filtered_df['occupancy_at_end_time'], label='Original Occupancy at End Time', marker='x', linestyle='-', alpha=0.5)

# Plot smoothed and trimmed data
plt.plot(trimmed_time_intervals, trimmed_smoothed_total_devices, label='trimmed_Smoothed Total Devices', marker='o', linestyle='-')
plt.plot(trimmed_time_intervals, trimmed_smoothed_occupancy_at_end_time, label='trimmed_Smoothed Occupancy at End Time', marker='x', linestyle='-')

plt.xlabel('Time Interval End')
plt.ylabel('Count')
plt.title('Smoothed Total Devices and Occupancy Over Time')
plt.legend()
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()