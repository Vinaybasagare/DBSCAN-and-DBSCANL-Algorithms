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

def merge_overlapping_clusters(clusters):
    """Merge clusters only when all elements in one cluster match all elements in another cluster."""
    merged_clusters = []

    while clusters:
        first, *rest = clusters
        first = set(first)

        rest2 = []
        for r in rest:
            r_set = set(r)
            # Merge only if one cluster is a subset of the other
            if first == r_set:
                first |= r_set
            else:
                rest2.append(r)

        merged_clusters.append(list(first))
        clusters = rest2

    return merged_clusters

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union

# Define the range_query_jaccard function
def range_query_jaccard(df, target_mac, cluster_macs, dr_threshold=1):
    target_data = df.loc[df['src'] == target_mac, 'dot11elt']
    if target_data.empty:
        return []
    target_data_rates = set(target_data.values[0])
    def calculate_jaccard(dr):
        if isinstance(dr, list) and dr:
            return jaccard_similarity(target_data_rates, set(dr))
        else:
            return 0
    df_cluster = df[df['src'].isin(cluster_macs)].copy()
    df_cluster['jaccard_similarity'] = df_cluster['dot11elt'].apply(calculate_jaccard)
    final_neighbors = df_cluster[df_cluster['jaccard_similarity'] >= dr_threshold]['src'].tolist()
    if target_mac not in final_neighbors:
        final_neighbors.append(target_mac)
    return final_neighbors

def range_query_rssi(target_mac, df, rssi_eps, min_samples):
    target_rssi = df.loc[df['src'] == target_mac, 'rssi'].values[0]
    rssi_mask = (df['rssi'] >= target_rssi - rssi_eps) & (df['rssi'] <= target_rssi + rssi_eps)
    rssi_neighbors = df[rssi_mask]
    neighbors = [target_mac] + rssi_neighbors['src'].tolist()
    neighbors = list(set(neighbors))
    return neighbors if len(neighbors) >= min_samples else []

def filter_clusters(clusters, min_samples):
    return [cluster for cluster in clusters if len(cluster) >= min_samples]

start_time = df['datetime'].min()
end_time = df['datetime'].max()
start_time = pd.to_datetime(start_time)
end_time = pd.to_datetime(end_time)
interval_duration = timedelta(minutes=10)
step = timedelta(minutes=5)

intervals = []
current_start = start_time
while current_start + interval_duration <= end_time:
    current_end = current_start + interval_duration
    intervals.append((current_start, current_end))
    current_start += step

min_samples = 3
results = []
winsize = 10
probes = 4
for start, end in intervals:
    group = df_fake[(df_fake['datetime'] >= start) & (df_fake['datetime'] < end)]
    if group.empty:
        continue

    num_real_devices = df_real[(df_real['datetime'] >= start) & (df_real['datetime'] < end)]['src'].nunique()
    max_occupancy = group['occupancy'].max()

    closest_record = group.iloc[-1]
    occupancy_at_end_time = closest_record['occupancy']
    all_neighbors = {}
    for mac in group['src'].unique():
        neighbors = range_query_rssi(mac, group, rssi_eps=4, min_samples=8)
        if neighbors:
            all_neighbors[mac] = neighbors
    unique_clusters = set()
    for mac, neighbors in all_neighbors.items():
        if neighbors:
            cluster = frozenset(neighbors)
            unique_clusters.add(cluster)
    final_clusters = [list(cluster) for cluster in unique_clusters]
    result_clusters = []
    for cluster in final_clusters:
        cluster_macs = cluster
        cluster_results = []
        for mac in cluster_macs:
            neighbors = range_query_jaccard(group, mac, cluster_macs, dr_threshold=1)
            if len(neighbors) >= min_samples:
                cluster_results.append(neighbors)
        result_clusters.extend(cluster_results)

    merged_clusters = merge_overlapping_clusters(result_clusters)
    final_merged_clusters = filter_clusters(merged_clusters, min_samples)
    unique_devices_count_fake = len(final_merged_clusters)
   # total_devices = (unique_devices_count_fake + num_real_devices)/(winsize*probes)
    total_devices = (unique_devices_count_fake + num_real_devices)
    results.append({
        'interval_start': start,
        'interval_end': end,
        'unique_devices_fake': unique_devices_count_fake,
        'num_real_devices': num_real_devices,
        'occupancy_at_end_time': occupancy_at_end_time,
        'max_occupancy': max_occupancy,
        'total_devices': total_devices
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)
print(len(results_df))
# Calculate moving average and absolute difference
window_size = 20
results_df['smoothed_total_devices1'] = results_df['total_devices'].rolling(window=window_size).mean()
results_df['absolute_difference1'] = abs(results_df['smoothed_total_devices1'] - results_df['occupancy_at_end_time'])

# Calculate the moving average and MAE
mae1 = results_df['absolute_difference1'].mean()

# Print the MAE
print(f"Mean Absolute Error (MAE1) smoothed: {mae1:.2f}")

# Filter design parameters
filter_order = 20 # Order of the filter
cutoff_frequency = 0.1  # Normalized cutoff frequency (0.1 means 10% of the Nyquist frequency)

# Design the FIR filter using a Hamming window
fir_coefficients = firwin(filter_order, cutoff_frequency, window='hamming')

# Smooth the total_devices data
smoothed_total_devices = filtfilt(fir_coefficients, 1.0, results_df['total_devices'])
# Smooth the occupancy_at_end_time data
smoothed_occupancy_at_end_time = filtfilt(fir_coefficients, 1.0, results_df['occupancy_at_end_time'])

# Trim points to handle edge effects
trim_points = filter_order // 2

# Trim the smoothed data
trimmed_smoothed_total_devices = smoothed_total_devices[trim_points:-trim_points]
trimmed_smoothed_occupancy_at_end_time = smoothed_occupancy_at_end_time[trim_points:-trim_points]

# Also trim the corresponding time intervals to match the length of the trimmed data
trimmed_time_intervals = results_df['interval_end'][trim_points:-trim_points]

# Trim the original data to match the length of the smoothed and trimmed data
trimmed_original_total_devices = results_df['total_devices'][trim_points:-trim_points]
trimmed_original_occupancy_at_end_time = results_df['occupancy_at_end_time'][trim_points:-trim_points]

# Calculate MAE for total devices
mae_total_devices = mean_absolute_error(trimmed_smoothed_total_devices, trimmed_smoothed_occupancy_at_end_time)

# Print the MAE values
print(f'MAE2E for Total Devices: {mae_total_devices}')
mse_total_devices = mean_squared_error(trimmed_smoothed_total_devices, trimmed_smoothed_occupancy_at_end_time)

# Root Mean Squared Error (RMSE)
rmse_total_devices = np.sqrt(mse_total_devices)
# Mean Absolute Percentage Error (MAPE)
#mape_total_devices = mean_absolute_percentage_error(trimmed_smoothed_total_devices, trimmed_smoothed_occupancy_at_end_time)
def smape(A, F):
    return 100 * np.mean(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

# Handle cases where the actual value is zero to avoid division by zero errors
# Replacing zero with a small number to avoid division by zero issues
epsilon = 1e-10
trimmed_smoothed_total_devices = np.where(trimmed_smoothed_total_devices == 0, epsilon, trimmed_smoothed_total_devices)
trimmed_smoothed_occupancy_at_end_time = np.where(trimmed_smoothed_occupancy_at_end_time == 0, epsilon, trimmed_smoothed_occupancy_at_end_time)

# Calculate sMAPE
smape_value = smape(trimmed_smoothed_occupancy_at_end_time, trimmed_smoothed_total_devices)

# Print the additional metrics
print(f'RMSE for Total Devices: {rmse_total_devices}')
print(f'MSE for Total Devices: {mse_total_devices}')
print(f'smape_value for Total Devices: {smape_value}')

# Plot the results
# plt.figure(figsize=(12, 6))
# plt.plot(trimmed_time_intervals, trimmed_original_total_devices, label='Original Total Devices')
# plt.plot(trimmed_time_intervals, trimmed_smoothed_total_devices, label='Smoothed Total Devices', linestyle='--')
# plt.plot(trimmed_time_intervals, trimmed_original_occupancy_at_end_time, label='Original Occupancy at End Time')
# plt.plot(trimmed_time_intervals, trimmed_smoothed_occupancy_at_end_time, label='Smoothed Occupancy at End Time', linestyle='--')
# plt.legend()
# plt.xlabel('Time Interval')
# plt.ylabel('Count')
# plt.title('Comparison of Total Devices and Occupancy Over Time')
# plt.show()
