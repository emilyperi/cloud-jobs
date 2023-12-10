import matplotlib.pyplot as plt
import numpy as np


def visualize_sar_image(img, i):
    try:

        # Plotting
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Visualize VV band
        vv_band = img[:, :, 0]
        ax[0].imshow(vv_band, cmap='cubehelix')
        ax[0].set_title('VV Band')
        ax[0].axis('off')

        # Visualize VH band
        vh_band = img[:, :, 1]
        ax[1].imshow(vh_band, cmap='cubehelix')
        ax[1].set_title('VH Band')
        ax[1].axis('off')

        fig.suptitle(f"Normalized Image Visualization", fontsize=12)

        plt.savefig(f"sample_validation_image{i}.png")
        plt.show()
    except Exception as e:
        print(f"Visualization failed {str(e)}")


validation_img = np.load("./data/s1_data/k-folds/validation/val_images_0.npy")
sample_size = 10  # Define your sample size
sampled_indices = np.random.choice(validation_img.shape[0], sample_size, replace=False)
sampled_images = validation_img[sampled_indices]
for i in range(sampled_images.shape[0]):
    img = sampled_images[i]
    visualize_sar_image(img, i)
