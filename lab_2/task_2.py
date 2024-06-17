import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.decomposition import PCA
from skimage import color


def apply_pca(image_bw: np.ndarray, variance_threshold=0.95):
    pca = PCA(variance_threshold)
    transformed_data = pca.fit_transform(image_bw)
    return pca, transformed_data


def reconstruct_image(pca: PCA, transformed_data: np.ndarray):
    reconstructed_image = pca.inverse_transform(transformed_data)
    return reconstructed_image


def plot_cumulative_variance(cumulative_variance, num_components):
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_variance)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by the Components')
    plt.axhline(y=0.95, color='r', linestyle='--')
    plt.axvline(x=num_components, color='k', linestyle='--')
    plt.grid(True)
    plt.show()


def plot_reconstructed_images(image_bw, n_components_list):
    plt.figure(figsize=(15, 10))
    for i, n_components in enumerate(n_components_list, start=1):
        pca = PCA(n_components)
        transformed_data = pca.fit_transform(image_bw)
        reconstructed_image = reconstruct_image(pca, transformed_data)

        plt.subplot(2, 3, i)
        plt.title(f"Components: {n_components}")
        plt.imshow(reconstructed_image, cmap='gray')
        plt.axis('on')  # Ensure the tick labels are displayed
    plt.tight_layout()
    plt.show()


# Load the image
image_path = os.path.join(os.path.dirname(__file__), 'new_york.jpeg')
image_raw = imread(image_path)
print(f"Original image shape: {image_raw.shape}")

# Point 0.25: Convert the image to grayscale
image_gray = color.rgb2gray(image_raw)
print(f"Grayscale image shape: {image_gray.shape}")
print(f"Grayscale max value: {image_gray.max()} (0.25 points)")

# Display original and grayscale images side by side
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_raw)
plt.title(f'Original Image ({image_raw.shape[0]}x{image_raw.shape[1]})')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(image_gray, cmap='gray')
plt.title(f'Grayscale Image ({image_gray.shape[0]}x{image_gray.shape[1]})')
plt.axis('off')
plt.show()

# Point 0.75: Apply PCA
pca, transformed_data = apply_pca(image_gray)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
num_components = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Number of components needed for 95% variance: {num_components} (0.75 points)")

# Point 0.5: Plot the cumulative variance graph
plot_cumulative_variance(cumulative_variance, num_components)
print("Displayed cumulative variance graph (additional 0.5 points)")

# Point 1: Reconstruct the image using the number of components needed for 95% variance
reconstructed_image_95 = reconstruct_image(pca, transformed_data)
plt.figure(figsize=(6, 6))
plt.imshow(reconstructed_image_95, cmap='gray')
plt.title(f'Reconstructed Image ({num_components} components, {image_gray.shape[0]}x{image_gray.shape[1]})')
plt.axis('on')
plt.show()
print("Displayed reconstructed image using the number of components for 95% variance (1 point)")

# Point 2: Reconstruct and display images using different numbers of components
n_components_list = [5, 15, 25, 75, 100, 170]
plot_reconstructed_images(image_gray, n_components_list)
print("Reconstructed and displayed images using different numbers of components (2 points)")
