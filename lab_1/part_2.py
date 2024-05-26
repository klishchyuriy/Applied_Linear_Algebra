import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_cv_image(image, title="Image"):

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_cv_images(original_image, transformed_image, title="Image Comparison"):

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Transformed Image')
    axes[1].axis('off')

    plt.suptitle(title)
    plt.show()


def create_canvas(objects, shape=(500, 500, 3)):

    canvas = np.zeros(shape, dtype=np.uint8)
    for obj, color in objects:
        pts = np.array(obj, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
    return canvas



object1 = np.array([[50, 450], [50, 350], [150, 450], [50, 450]])  # Triangle
object2 = np.array([[200, 400], [300, 400], [350, 450], [250, 450], [200, 400]])  # Trapeze

# Create canvas and plot original objects
canvas = create_canvas([(object1, (255, 0, 0)), (object2, (0, 255, 0))])
plot_cv_image(canvas, "Original Objects")


def apply_transformation(canvas, matrix):
    rows, cols = canvas.shape[:2]
    transformed_image = cv2.warpAffine(canvas, matrix, (cols, rows))
    return transformed_image


# Rotation
angle = 30  # Change the angle to 30 degrees
rotation_matrix = cv2.getRotationMatrix2D((250, 250), angle, 1)
rotated_image = apply_transformation(canvas, rotation_matrix)
plot_cv_images(canvas, rotated_image, "Rotated by 30 degrees using OpenCV")

# Scaling
scale_x, scale_y = 1.5, 0.7  # Scale x by 1.5, y by 0.7
scaling_matrix = np.float32([[scale_x, 0, 0], [0, scale_y, 0]])
scaled_image = apply_transformation(canvas, scaling_matrix)
plot_cv_images(canvas, scaled_image, f"Scaled Image by factors {scale_x} and {scale_y} using OpenCV")

# Reflection
reflected_image = cv2.flip(canvas, 1)  # Flip around Y-axis
plot_cv_images(canvas, reflected_image, "Reflected Image along Y-axis")

# Shear
shear_factor = 0.3  # Shear along x-axis with a factor of 0.3
shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])
sheared_image = apply_transformation(canvas, shear_matrix)
plot_cv_images(canvas, sheared_image, f"Sheared Image along X-axis with factor {shear_factor}")

# Custom Transformation
custom_matrix = np.float32([[1, 0.5, 0], [-0.5, 1, 0]])
custom_transformed_image = apply_transformation(canvas, custom_matrix)
plot_cv_images(canvas, custom_transformed_image, "Custom Transformed Image")

image_path = '/Users/klishchyuriy/MyProjects/like/Applied_Linear_Algebra/lab_1/math_guy.jpeg'
image = cv2.imread(image_path)
if image is not None:
    plot_cv_image(image, "Original Image")

    # Apply transformations
    rotated_real_image = apply_transformation(image, rotation_matrix)
    sheared_real_image = apply_transformation(image, shear_matrix)
    reflected_real_image = cv2.flip(image, 1)

    # Plot the results
    plot_cv_images(image, rotated_real_image, "Image Rotated by 30 Degrees")
    plot_cv_images(image, sheared_real_image, "Sheared Real Image")
    plot_cv_images(image, reflected_real_image, "Reflected Real Image along Y-axis")
else:
    print("Failed to load image. Check the file path.")