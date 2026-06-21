#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def timestamp_converter(x):
    s = str(x, 'utf-8')
    print(f"Converting timestamp: {s}")
    if s == "timestamp":
        return 0.0
    else:
        return float(s)

def compute_avg_theta(data: np.ndarray):
    jumps = np.diff(data)

    # Handle 360-degree wraparound
    jumps[jumps < 0] += 360.0

    return np.mean(jumps)
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

    return  avgs, std, coefVariation, outlier_percentage     

def plot_precision(angles: np.ndarray, quality: np.ndarray, distances: np.ndarray, ax1, ax2, ax3, ax4, bin_res: float = 1.0):
    # Compute bins:
    bins = np.arange(0, 360 + bin_res, bin_res)
    
    stats_per_bin = np.zeros((len(bins) - 1, 4))  # columns: avg, std, coefVariation, outlier_percentage
    quality_per_bin = np.zeros(len(bins) - 1)
    
    for bin_start, bin_end in zip(bins[:-1], bins[1:]):
        # Get indices of angles that fall into the current bin
        indices = np.where((angles >= bin_start) & (angles < bin_end))[0]
        if len(indices) > 0:
            quality_per_bin[int(bin_start // bin_res)] = 1 - (np.sum(quality[indices] == 0) / len(indices))
            avgs, std, coefVariation, outlier_percentages = compute_boxplot_stats(distances[indices])
            stats_per_bin[int(bin_start // bin_res)] = [avgs, std, coefVariation, outlier_percentages]
        else:
            quality_per_bin[int(bin_start // bin_res)] = 0
            stats_per_bin[int(bin_start // bin_res)] = [0, 0, 0, 0]
   
    # Create polar plot 
    thetaAxis = np.deg2rad(bins[:-1])  # Convert bin edges to radians for polar plot

    theta = thetaAxis
    r_mean = stats_per_bin[:, 0]
    r_std  = stats_per_bin[:, 1]
    r_lower = np.maximum(r_mean - r_std, 0) 

    # Close the loop so fill wraps around 360°
    theta_closed = np.concatenate([theta, theta[::-1]])
    r_closed     = np.concatenate([r_mean + r_std, r_lower[::-1]])

    ax1.fill(theta_closed, r_closed, color='blue', alpha=0.25)
    ax1.plot(theta, r_mean, color='blue', linewidth=1)
    
    # Plot boxplot of cov per bin
    ax2.bar(bins[:-1], stats_per_bin[:, 2],
        width=bin_res,
        color='tomato')
    ax3.bar(bins[:-1], 
                stats_per_bin[:, 3], 
                width=bin_res, 
                align='edge')
    ax4.plot(bins[:-1], 
             quality_per_bin, 
             marker='o')

    # Replace these in plot_precision for ax1 (the polar one):
    ax1.set_title("Radial Average Distance ± Std Dev")
    ax1.set_theta_zero_location("N")
    ax1.set_theta_direction(-1)
    # Remove ax1.set_xlabel and ax1.set_ylabel — they don't render on polar axes

    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Coefficient of Variation')
    ax2.set_title('LiDAR Coefficient of Variation per Angle Bin')
    ax2.set_ylim(0, np.percentile(stats_per_bin[:, 2], 95))
    ax2.grid()

    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Outlier Percentage')
    ax3.set_ylim(0, 1)  # Set y-axis limits to [0, 1] for percentage
    ax3.set_title('LiDAR Outlier Percentage per Angle Bin')
    ax3.grid()

    
    ax4.set_xlabel('Angle (degrees)')
    ax4.set_ylabel('Quality')
    ax4.set_title('LiDAR Quality per Angle Bin')
    ax4.grid()

    return bins, stats_per_bin, quality_per_bin

    

if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Plot LiDAR quality data.")
    parser.add_argument("filename", type=str, help="Path to the data file")
    parser.add_argument("binRes", type=float, help="Bin resolution for angle bins (e.g., 1.0 for 1 degree bins)")
    parser.add_argument("outputFile", type=str, required=False,  help="Path to save bin statistics CSV")

    args = parser.parse_args()
    # Load data from file
    df = pd.read_csv(args.filename, delimiter=",")

    print("Average theta jump: ", compute_avg_theta(df["theta"].to_numpy()))


    # Create ONE window for all plots
    fig = plt.figure(figsize=(15, 10))

    ax_avg    = fig.add_subplot(2, 2, 1, projection='polar')  # averaged distance — polar
    ax_cov    = fig.add_subplot(2, 2, 2)                       # CoV bar chart
    ax_out    = fig.add_subplot(2, 2, 3)                       # outlier % bar chart
    ax_qual   = fig.add_subplot(2, 2, 4)                       # quality line plot

    bins, stats, quality = plot_precision(df["theta"].to_numpy(), df["quality"].to_numpy(), 
                df["dist_mm"].to_numpy(), 
                ax_avg, ax_cov, ax_out, ax_qual, args.binRes)

    fig2 = plt.figure(figsize=(7, 7))
    ax_noise = fig2.add_subplot(111, projection='polar')
    sc = ax_noise.scatter(
        np.deg2rad(df["theta"]),
        df["dist_mm"],
        c=df["quality"],
        cmap="RdYlGn",
        s=3,
        alpha=0.1
    )
    ax_noise.set_theta_zero_location("N")
    ax_noise.set_theta_direction(-1)

    # Show the plots
    plt.tight_layout()
    plt.show()


