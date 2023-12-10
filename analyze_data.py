from typing import List

import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
from shapely.geometry import Point
from skimage import exposure
from affine import Affine
import mplleaflet
import math
import cv2


def string_to_affine(s):
    # Extract values from the string
    values = [float(v) for row in s.split("\n") for v in row.strip("| ").split(",")]

    # Construct and return the Affine object
    return Affine(*values)


def get_center_point(row):
    # Extract the height and width from the row
    height, width = row['height'], row['width']

    # Calculate the center pixel coordinates
    center_row, center_col = height / 2, width / 2

    # Apply the affine transform to get the geographic coordinates
    x, y = string_to_affine(row['affine_transform']) * (center_col, center_row)

    # Return the Point geometry
    return Point(x, y)


def group_by_center_point(df: pd.DataFrame, debug=False):
    def rounded_point(point, decimals=1):
        if point:
            return round(point.x, decimals), round(point.y, decimals)
        return None

    df['rounded_center'] = df['center_point'].apply(rounded_point, decimals=2)
    grouped = df.groupby('rounded_center')
    gdfs = []
    for rounded_center, group_data in grouped:
        if rounded_center is None:
            continue
        unique_crs_values = group_data["crs"].nunique()
        if unique_crs_values == 1:
            gdfs.append(gpd.GeoDataFrame(group_data, geometry='center_point', crs=group_data["crs"].iloc[0]))
            print(f"All rows for center_point {rounded_center} have the same crs.") if debug else None
        else:
            print(f"Rows for center_point {rounded_center} have different crs values.") if debug else None
    return gdfs


def create_geo_dataframes(df: pd.DataFrame):
    grouped = df.groupby("crs")
    gdfs = []
    for crs, group_data in grouped:
        if crs != "None":
            gdf = gpd.GeoDataFrame(group_data, geometry='center_point', crs=crs)
            gdfs.append(gdf)
    return gdfs


def display_map_clusters(gdfs: List):
    all_lats, all_lons = [], []

    for gdf in gdfs:
        gdf_wgs84 = gdf.to_crs("EPSG:4326")
        all_lats.extend(gdf_wgs84.geometry.y.tolist())
        all_lons.extend(gdf_wgs84.geometry.x.tolist())

    mean_lat, mean_lon = sum(all_lats) / len(all_lats), sum(all_lons) / len(all_lons)
    print(f'Mean Lat {mean_lat}, Mean Lon {mean_lon}')

    # Initialize map
    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=6)  # Adjust zoom_start as necessary

    # Iterate over each GeoDataFrame and plot on map
    for gdf in gdfs:
        gdf_wgs84 = gdf.to_crs("EPSG:4326")

        # Using clustering to group close points
        cluster = MarkerCluster().add_to(m)

        for idx, row in gdf_wgs84.iterrows():
            folium.Marker([row['center_point'].y, row['center_point'].x]).add_to(cluster)

    return m


def display_map_scatter(gdf, save_path):
    gdf_wgs84 = gdf.to_crs("EPSG:4326")
    lats = gdf_wgs84['center_point'].y.tolist()
    lons = gdf_wgs84['center_point'].x.tolist()

    fig, ax = plt.subplots()
    ax.scatter(lons, lats, marker='o', c='red', s=10)

    mplleaflet.save_html(fig=fig, fileobj=save_path)


def average_statistics(np_array):
    print("Average Statistics Across All Images:")

    for channel, color in enumerate(['Red', 'Green', 'Blue']):
        channel_data = np_array[:, :, :, channel]

        min_value = np.min(channel_data)
        max_value = np.max(channel_data)
        mean_value = np.mean(channel_data)
        std_dev = np.std(channel_data)

        print(f"{color} Channel:")
        print(f"  Min Value: {min_value}")
        print(f"  Max Value: {max_value}")
        print(f"  Mean Value: {mean_value}")
        print(f"  Standard Deviation: {std_dev}")

    print("------")


def filter_zeros_indices(images: np.ndarray):
    not_all_zero_samples = np.any(images != 0, axis=(1, 2, 3))
    indices = np.where(not_all_zero_samples)[0]
    return indices


def equalize_rgb_image(image):
    # Equalize each channel separately
    eq_red = exposure.equalize_hist(image[:, :, 0])
    eq_green = exposure.equalize_hist(image[:, :, 1])
    eq_blue = exposure.equalize_hist(image[:, :, 2])

    # Stack the channels back together
    return np.stack([eq_red, eq_green, eq_blue], axis=-1)


def display_images(n_images, rgb_images, captions=None, save_path=None):
    # Calculate the number of rows needed
    rows = math.ceil(n_images / 4)

    # Create a figure with a grid of subplots
    fig, axs = plt.subplots(rows, 4, figsize=(15, 5 * rows))

    # Handle the case where there's only one row, which means axs is a 1D array
    if rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for i in range(n_images):
        row = i // 4
        col = i % 4
        img = rgb_images[i]
        if captions:
            axs[row, col].set_title(captions[i])
        axs[row, col].imshow(img)
        axs[row, col].axis('off')  # To hide axes

    # Turn off any remaining empty subplots
    for i in range(n_images, rows * 4):
        row = i // 4
        col = i % 4
        axs[row, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    #plt.show()


def compute_histogram_rgb(image, range, bins=256):
    # For each channel, compute histogram and return all three histograms
    red_hist = np.histogram(image[..., 0], bins=bins, range=range)[0]
    green_hist = np.histogram(image[..., 1], bins=bins, range=range)[0]
    blue_hist = np.histogram(image[..., 2], bins=bins, range=range)[0]

    return red_hist, green_hist, blue_hist


def plot_histogram_rgb(rgb_images):
    red_histograms = []
    green_histograms = []
    blue_histograms = []

    for image in rgb_images:
        red_hist, green_hist, blue_hist = compute_histogram_rgb(image)
        red_histograms.append(red_hist)
        green_histograms.append(green_hist)
        blue_histograms.append(blue_hist)

    # Average histograms for each channel
    average_red_histogram = np.mean(red_histograms, axis=0)
    average_green_histogram = np.mean(green_histograms, axis=0)
    average_blue_histogram = np.mean(blue_histograms, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(average_red_histogram, color='red', label='Red Channel')
    plt.plot(average_green_histogram, color='green', label='Green Channel')
    plt.plot(average_blue_histogram, color='blue', label='Blue Channel')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Average Frequency')
    plt.title('Average Histogram Across All Images')
    plt.legend()
    plt.show()


def scale_16bit_images_basic(images: np.ndarray):
    scaled_images = []

    for img in images:
        scaled_img = (img - img.min()) / (img.max() - img.min()) * 255.0
        scaled_images.append(scaled_img)

    scaled_images = np.array(scaled_images, dtype=np.uint8)
    return scaled_images


def scale_16bit_images_log(images: np.ndarray):
    scaled_images = []
    for img in images:
        c = 255 / np.log(1 + np.max(img))
        log_transformed = c * np.log(1 + img)
        scaled_images.append(log_transformed)

    scaled_images = np.array(scaled_images, dtype=np.uint8)
    return scaled_images

def max_image_value(images: np.array):
    return images.max()
def scale_16bit_images_cv2(images: np.ndarray, range):
    scaled_images = []
    for img in images:
        channels = cv2.split(img)  # Split image into its RGB channels
        result_channels = []
        for channel in channels:
            channel_8bit = cv2.convertScaleAbs(channel, alpha=(255.0 / range))
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))
            clahe_img = clahe.apply(channel_8bit)
            result_channels.append(clahe_img)

        merged_img = cv2.merge(result_channels)  # Merge the processed channels back into an RGB image
        scaled_images.append(merged_img)

    scaled_images = np.array(scaled_images, dtype=np.uint8)
    return scaled_images


if __name__ == '__main__':
    rgb_images_np = np.load("data/processed-data/rgb_img.npy")
    filter_indices = filter_zeros_indices(rgb_images_np)
    rgb_images_np_filtered = rgb_images_np[filter_indices]

    s2_labels = np.load("data/processed-data/s2_labels.npy")
    s2_filtered_labels = np.load("data/processed-data/s2_labels_filtered.npy")
    s2_coverage_labels = np.load("data/processed-data/s2_coverage_labels.npy")
    s2_flooding_and_coverage = np.column_stack((s2_labels, s2_coverage_labels))

    captions = [f'Flooding Label: {value[0]}, Full Coverage: {value[1]}' for value in s2_flooding_and_coverage]
    captions_filtered = [captions[i] for i in filter_indices]


    # Scaling types
    rgb_images_scaled_basic = scale_16bit_images_basic(rgb_images_np)
    np.save("data/processed-data/scaled_basic_rgb_images.npy", rgb_images_scaled_basic)

    # rgb_images_scaled_log = scale_16bit_images_log(rgb_images_np)
    # np.save("scaled_log_rgb_images.npy", rgb_images_scaled_log)

    rgb_images_filtered_scaled_basic = rgb_images_scaled_basic[filter_indices]
    np.save("data/processed-data/scaled_basic_rgb_images_filtered.npy", rgb_images_filtered_scaled_basic)

    # rgb_images_filtered_scaled_log = rgb_images_scaled_log[filter_indices]
    # np.save("scaled_log_rgb_images_filtered.npy", rgb_images_filtered_scaled_log)

    # max_value = max_image_value(rgb_images_np)
    # print(max_value)
    #
    # rgb_images_scaled_clahe = scale_16bit_images_cv2(rgb_images_np, max_value)
    # np.save("scaled_clahe_rgb_images.npy", rgb_images_scaled_clahe)
    #
    # rgb_images_filtered_scaled_clahe = rgb_images_scaled_clahe[filter_indices]
    # np.save("scaled_clahe_rgb_images_filtered.npy", rgb_images_filtered_scaled_clahe)

    display_images(8, rgb_images_scaled_basic, captions=captions[0:50], save_path="plots/rgb_images_scaled_basic.png")
    display_images(8, rgb_images_filtered_scaled_basic, captions=captions_filtered[0:50],
                   save_path="plots/rgb_images_filtered_basic_basic.png")
