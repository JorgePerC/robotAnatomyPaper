import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from matplotlib.widgets import Slider
import re

def timestamp_converter(x):
    s = str(x, 'utf-8')
    print(f"Converting timestamp: {s}")
    if s == "timestamp":
        return 0.0
    else:
        return float(s)

def compute_avg_theta(dataFrame: pd.DataFrame, window_seconds: float, voltage=None):
    df = dataFrame.copy()

    df["t_continuous"] = df["t_continuous"] - df["t_continuous"].min()

    max_time = df["t_continuous"].max()
    window_ms = window_seconds*1000

    # Build time bins
    time_bins = np.arange(0, max_time + window_ms, window_ms)
    n_bins = len(time_bins) - 1

    stats_per_bin = np.full((n_bins, 3), np.nan)  # mean, std, max

    for i, (t_start, t_end) in enumerate(zip(time_bins[:-1], time_bins[1:])):
        indices = np.where(
            (df["t_continuous"] >= t_start) & (df["t_continuous"] < t_end)
        )[0]

        if len(indices) < 50:
            continue

        angles = df["theta"].iloc[indices].to_numpy()
        jumps  = np.diff(angles)
        # We could remove zeros
        # This would be indicative of duplicated secuential points
        # But I'll keep them for pure statistical analysis, as they do represent a real phenomenon
        # jumps = jumps[jumps > 0] 
        # Handle 360-degree wraparound
        jumps[jumps < 0] += 360.0
        jumps  = jumps[jumps < 180]

        if len(jumps) == 0:
            continue

        stats_per_bin[i] = [
            np.mean(jumps),
            np.std(jumps),
            np.max(jumps)
        ]

        bin_centres = (time_bins[:-1] + time_bins[1:]) / 2000

    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Angular Resolution Stability Over Time — {voltage}V", fontsize=13)

    labels = ["Mean Step (°)", "Std Step (°)", "Max Gap (°)"]
    colors = ["steelblue", "tomato", "seagreen", "mediumpurple"]

    for ax, col, label, color in zip(axes, range(4), labels, colors):
        ax.plot(bin_centres, stats_per_bin[:, col],
                marker='o', linewidth=1.5, color=color)
        ax.axhline(np.mean(stats_per_bin[:, col]), color='grey',
                   linestyle='--', alpha=0.5,
                   label=f'mean: {np.mean(stats_per_bin[:, col]):.4f}°')
        ax.set_ylabel(label)
        ax.legend(loc='upper right')
        ax.grid(True)

    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    # --- t_continuous diagnostics ---
    fig_ts, ax_ts = plt.subplots(1, 1, figsize=(10, 3))

    ax_ts.plot(df["t_continuous"] / 1000, np.diff(df["t_continuous"], prepend=df["t_continuous"].iloc[0]),
            linewidth=0.5)

    ax_ts.set_title(f"Inter-sample Timestamp Differences — {voltage}V")
    ax_ts.set_xlabel("Time (seconds)")
    ax_ts.set_ylabel("Δt (ms)")
    ax_ts.grid(True)
    plt.tight_layout()
    

def compute_rotation_offset(angles, distances, 
                             edge_dist_min=130, edge_dist_max=160,
                             reference_angle=0.0):
    """
    Finds the midpoint angle of the flat top edge and returns
    the rotation needed to align it to reference_angle.
    """
    # Get all points that are on the flat top edge
    mask = (distances >= edge_dist_min) & (distances <= edge_dist_max)
    edge_angles = angles[mask]

    if len(edge_angles) == 0:
        print("No edge points found — adjust edge_dist_min/max")
        return 0.0

    # Handle wraparound by converting to unit vectors and averaging
    angles_rad = np.deg2rad(edge_angles)
    mean_x = np.mean(np.cos(angles_rad))
    mean_y = np.mean(np.sin(angles_rad))
    mean_angle = np.rad2deg(np.arctan2(mean_y, mean_x)) % 360

    offset = (reference_angle - mean_angle) % 360
    print(f"Edge midpoint at {mean_angle:.2f}°, rotation offset: {offset:.2f}°")
    return offset

def shift_angles(angles, shift_deg):
    return (angles + shift_deg) % 360

def compute_boxplot_stats(binData):
    outlier_percentage = 0
    coefVariation = 0
    avgs = 0

    if len(binData) == 0:
        outlier_count = 0
        coefVariation = 0
        avgs = 0
        return avgs, coefVariation, outlier_count

    # average
    avgs = np.mean(binData)

    # standard deviation
    std = np.std(binData)
    coefVariation = std / (np.mean(binData) + 1e-6)
    

    # quartiles
    q1 = np.percentile(binData, 25)
    q3 = np.percentile(binData, 75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    # outliers
    outliers = binData[(binData < lower) | (binData > upper)]
    outlier_percentage = len(outliers) / len(binData) 

    return avgs, std, coefVariation, outlier_percentage

def plot_precision(dataFrame: pd.DataFrame,     
                   bin_res, angle_min=0, angle_max=360):

    angles = dataFrame["theta"].to_numpy().copy()
    quality = dataFrame["quality"].to_numpy().copy()
    distances = dataFrame["dist_mm"].to_numpy().copy()
    # Compute bins:
    bins = np.arange(angle_min, angle_max + bin_res, bin_res)

    stats_per_bin = np.zeros((len(bins) - 1, 4))  # columns: avg, std, coefVariation, outlier_percentage
    quality_per_bin = np.zeros(len(bins) - 1)
    
    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        indices = np.where((angles >= bin_start) & (angles < bin_end))[0]
        idxVal = int(bin_start // bin_res) - int(angle_min // bin_res)  # index for storing results
        if len(indices) > 0:
            valid_mask = quality[indices] != 0
            valid_indices = indices[valid_mask]  # indices into original arrays
            
            quality_per_bin[idxVal] = len(valid_indices) / len(indices)
            
            if len(valid_indices) > 0:
                avgs, std, coefVariation, outlier_rate = compute_boxplot_stats(distances[valid_indices])
                stats_per_bin[idxVal] = [avgs, std, coefVariation, outlier_rate]
            else:
                stats_per_bin[idxVal] = [0, 0, 0, 0]
        else:
            quality_per_bin[idxVal] = 0
            stats_per_bin[idxVal] = [0, 0, 0, 0]

    # Create polar plot 
    thetaAxis = np.deg2rad(bins[:-1])  # Convert bin edges to radians for polar plot
    
    r_mean = stats_per_bin[:, 0]
    r_std  = stats_per_bin[:, 1]
    r_lower = np.maximum(r_mean - r_std, 0) 

    # Close the loop so fill wraps around 360°
    theta_closed = np.concatenate([thetaAxis, thetaAxis[::-1]])
    r_closed     = np.concatenate([r_mean + r_std, r_lower[::-1]])

    # ---- AX1: Polar Plot ----
    ax_avg.fill(theta_closed, r_closed, color='blue', alpha=0.25)
    ax_avg.plot(thetaAxis, r_mean, color='blue', linewidth=1)
    
    ax_avg.set_thetagrids(np.arange(0, 360, 15))
    ax_avg.set_title("Radial Average Distance ± Std Dev")
    ax_avg.set_theta_zero_location("N")
    ax_avg.set_theta_direction(-1)
    
    # ---- AX2: CV Bar Plot ----
    # corners = [(45, 50),   # corner 1 — e.g. (28, 48)
    #         (145, 150),   # corner 2
    #         (205, 220),   # corner 3
    #         (313, 318)   # corner 4
    #     ]
    
    # # Clean mask
    # clean_mask = get_border_mask(bins, corners) & (stats_per_bin[:, 0] > 50)
    # stats_per_bin[~clean_mask, 2] = 0
    
    # # ax_cov.bar(bins[:-1], stats_per_bin[:, 2], width=bin_res, color='tomato', align='edge')
    # # ax_cov.set_xlabel('Angle (degrees)')
    # # ax_cov.set_ylabel('Coefficient of Variation (log scale)')
    # # ax_cov.set_title('LiDAR Coefficient of Variation per Angle Bin')
    # # ax_cov.set_yscale('log')
    # # ax_cov.set_ylim(1e-3, 1e-1)
    # # ax_cov.autoscale(axis='x', tight=True)
    # # ax_cov.grid(True, which='both')  # 'both' shows minor grid lines too, useful for log scale

    mask = (dataFrame["quality"] > 0) & (dataFrame["dist_mm"] > 0)
    valid = dataFrame[mask]

    # Map each point's angle to its bin's CoV
    bin_indices = np.digitize(valid["theta"].to_numpy(), bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(stats_per_bin) - 1)
    point_std   = stats_per_bin[bin_indices, 1]

    sc = ax_cov.scatter(valid["theta"], valid["dist_mm"],
                    c=point_std,
                    cmap='RdYlGn_r',   # red = high std, green = low std
                    s=2,
                    alpha=0.4,
                    norm=plt.Normalize(vmin=np.percentile(point_std, 5),
                                       vmax=np.percentile(point_std, 95)))

    # Overlay bin mean line
    has_data = stats_per_bin[:, 0] > 0
    ax_cov.plot(bins[:-1][has_data], stats_per_bin[has_data, 0],
            color='black', linewidth=1.5, alpha=0.6, label='Bin mean')

    cbar = plt.colorbar(sc, ax=ax_cov)
    cbar.set_label('Std Dev (mm)')

    ax_cov.set_xlabel('Angle (degrees)')
    ax_cov.set_ylabel('Distance (mm)')
    ax_cov.set_title('Angle vs Distance — coloured by Std Dev')
    ax_cov.legend(loc='upper right')
    ax_cov.grid(True)

    # ---- AX3: Quality Line Plot ----

    invalid_rate  = 1 - quality_per_bin                    # proportion with quality == 0
    valid_rate    = quality_per_bin                         # proportion with quality > 0
    outlier_rate  = stats_per_bin[:, 3] * valid_rate        # outliers as proportion of ALL readings
    clean_rate    = valid_rate - outlier_rate  

    ax_qual.bar(bins[:-1], clean_rate,   width=1, align='edge', 
           color='steelblue', label='Valid (clean)')
    ax_qual.bar(bins[:-1], outlier_rate, width=1, align='edge', 
           color='orange',    label='Valid (outlier)',
           hatch='///',       edgecolor='darkorange',
           bottom=clean_rate)
    ax_qual.bar(bins[:-1], invalid_rate, width=1, align='edge', 
           color='white',    label='Invalid',
                         edgecolor='tomato',
           bottom=clean_rate + outlier_rate) 
    
    ax_qual.legend(loc='lower center')
    ax_qual.set_xlabel('Angle (degrees)')
    ax_qual.set_ylabel('Proportion of readings')
    ax_qual.set_title('LiDAR Quality per Angle Bin')
    ax_qual.grid(True)
    
    # 2. Force layout clean up and draw the update
    fig_precision.tight_layout()


    return bins, stats_per_bin, quality_per_bin

def plot_raw(df: pd.DataFrame, timeRange ):
    ax_noise.cla()
    
    t_start = pd.to_datetime(timeRange[0]*1000, unit='s')
    t_end   = pd.to_datetime(timeRange[1]*1000, unit='s') #timeRange[1] #

    indices = np.where((df["timestamp"] >= t_start) & (df["timestamp"] < t_end))[0]

    print("Number of points in raw plot: ", len(indices))
    sc = ax_noise.scatter(
        np.deg2rad(df["theta"].iloc[indices]),
        df["dist_mm"].iloc[indices],
        s=3,
        alpha=0.1
    )
    ax_noise.set_theta_zero_location("N")
    ax_noise.set_theta_direction(-1)

def get_border_mask(bins, corner_ranges):
    """
    corner_ranges: list of (min_angle, max_angle) tuples to exclude
    """
    mask = np.ones(len(bins) - 1, dtype=bool)
    for a_min, a_max in corner_ranges:
        mask &= ~((bins[:-1] >= a_min) & (bins[:-1] < a_max))
    return mask
def make_continuous_time(df, gap_threshold_seconds=3):
    t = df["timestamp"].to_numpy()
    diffs = np.diff(t, prepend=t[0])
    
    # Estimate normal inter-sample interval from non-gap diffs
    normal_interval = np.median(diffs[diffs < gap_threshold_seconds])
    
    correction = 0
    t_continuous = t.copy().astype(float)
    
    for i in range(1, len(t)):
        if diffs[i] > gap_threshold_seconds*1000:
            # Only remove the excess, keep the normal interval
            correction += diffs[i] - normal_interval
        t_continuous[i] -= correction
    
    return t_continuous


def compute_distance_covariance(df: pd.DataFrame):
    
    # Use only valid data
    mask = (df["quality"] > 0) & (df["dist_mm"] > 0)
    valid_dist = df["dist_mm"][mask].to_numpy()
    valid_qual = df["quality"][mask].to_numpy()

    # Bin by distance
    dist_bins = np.arange(0, valid_dist.max() + 5, 5)  # 5mm bins
    cov_per_dist = []
    dist_centers = []
    point_counts = []  # add this

    for d_start, d_end in zip(dist_bins[:-1], dist_bins[1:]):
        in_bin = valid_dist[(valid_dist >= d_start) & (valid_dist < d_end)]
        if len(in_bin) > 5:  # minimum points to compute meaningful stats
            std = np.std(in_bin)
            mean = np.mean(in_bin)
            cov = std / (mean + 1e-6)
            cov_per_dist.append(cov)
            dist_centers.append((d_start + d_end) / 2)
            point_counts.append(len(in_bin))  # add this

    ax_distCov.plot(dist_centers, cov_per_dist, marker='o', color='tomato', linewidth=1.5)
    ax_distCov.set_xlabel('Distance (mm)')
    ax_distCov.set_ylabel('Coefficient of Variation', labelpad=100)
    ax_distCov.set_title('Measurement Consistency vs Distance')
    ax_distCov.grid(True)

    ax_count = ax_distCov.twinx()
    ax_count.bar(dist_centers, point_counts, width=4, alpha=0.15, color='grey', label='Point count')
    ax_count.set_ylabel('Points per bin', color='grey', labelpad=100)

def compute_overall_stats(bins, stats_per_bin, quality_per_bin, bin_res):
    """
    Uses per-bin stats already computed by plot_precision.
    stats_per_bin columns: avg, std, CoV, outlier_percentage
    quality_per_bin: fraction of valid readings per bin
    """
    # Estimate total readings per bin from quality data
    # quality_per_bin = valid/total, so we need total counts
    # Use bin width and assume uniform sampling as approximation
    
    has_data = stats_per_bin[:, 0] > 0

    # Weighted average of outlier percentage across bins with data
    outlier_pcts = stats_per_bin[has_data, 3]
    quality_pcts = quality_per_bin[has_data]
    invalid_pcts = 1 - quality_pcts

    print(f"Bins with data:              {has_data.sum()} / {len(bins) - 1}")
    print(f"Mean invalid rate:           {np.mean(invalid_pcts)*100:.2f}%")
    print(f"Mean outlier rate (of valid):{np.mean(outlier_pcts)*100:.2f}%")
    print(f"Max outlier rate:            {np.max(outlier_pcts)*100:.2f}% at {bins[:-1][has_data][np.argmax(outlier_pcts)]:.1f}°")
    print(f"Max invalid rate:            {np.max(invalid_pcts)*100:.2f}% at {bins[:-1][has_data][np.argmax(invalid_pcts)]:.1f}°")

    return {
        "mean_invalid_pct":  np.mean(invalid_pcts) * 100,
        "mean_outlier_pct":  np.mean(outlier_pcts) * 100,
        "max_outlier_pct":   np.max(outlier_pcts)  * 100,
        "max_invalid_pct":   np.max(invalid_pcts)  * 100,
    }
# =========== Main Code ===========
parser = argparse.ArgumentParser(description="Plot LiDAR quality data.")
parser.add_argument("filename", type=str, help="Path to the data file")
parser.add_argument("binRes", type=float, help="Bin resolution for angle bins (e.g., 1.0 for 1 degree bins)")
parser.add_argument("secondsStart", type=float, help="Start time in seconds for the time range")
parser.add_argument("secondsEnd", type=float, help="End time in seconds for the time range")
parser.add_argument("--outputFile", type=str, required=False,  help="Path to save bin statistics CSV")

args = parser.parse_args()

# Get voltage from filename (assuming format like "lidar_data_9V.csv")

match = re.search(r'(\d+)v', args.filename)
if match:
    voltage = match.group(1)
    print(f"Detected voltage: {voltage}V")
else:    
    voltage = "Unknown"
    print("Could not detect voltage from filename, defaulting to 'Unknown'")

# Load data from file
df = pd.read_csv(args.filename, delimiter=",")

# Remove gaps in sample time
df["t_continuous"] = make_continuous_time(df)

# Compute average theta per time window
#compute_avg_theta(df, window_seconds=20, voltage=voltage)


# Rotate
offset = compute_rotation_offset(df["theta"].to_numpy(), 
                                     df["dist_mm"].to_numpy(), reference_angle=315.0)  # Assuming the flat edge should be at 45 degrees

print("Offset", offset)
df["theta"] = (df["theta"] + offset) % 360


# Convert timestamp to seconds relative to start
df["t_continuous"] = df["t_continuous"] - df["t_continuous"].min()
df["t_continuous"] = pd.to_datetime(df["t_continuous"], unit='s')


fig_polar = plt.figure(figsize=(7, 7))
ax_avg    = fig_polar.add_subplot(111, projection='polar')


fig_precision = plt.figure(figsize=(15, 10))
fig_precision.suptitle(f"LiDAR deadzone Precision Analysis — {voltage}V", fontsize=16)
ax_cov  = fig_precision.add_subplot(2, 1, 1)
ax_qual = fig_precision.add_subplot(2, 1, 2, sharex=ax_cov)
 
angle_min = 145 #0
angle_max = 215 #360

angle_mask = (df["theta"] >= angle_min) & (df["theta"] <= angle_max)
df_window  = df[angle_mask]

bins, stats, quality = plot_precision(dataFrame=df_window, 
                                      bin_res=args.binRes,
                                      angle_min=angle_min, 
                                      angle_max=angle_max,)
compute_overall_stats(bins, stats, quality, args.binRes)
# Distance covariance
#compute_distance_covariance(df_window)
# Show graphs
plt.show()


if args.outputFile:
    # Save bin statistics to CSV
    bin_stats_df = pd.DataFrame(stats, columns=["Average", "StdDev", "CoV", "OutlierPercentage"])
    bin_stats_df["Quality"] = quality
    bin_stats_df["Bin"] = bins[:-1]
    bin_stats_df.to_csv(args.outputFile, index=False)

"""
At 9 volts, we see less CoV because we have less points
# We should compare distances rather than angles 

"""