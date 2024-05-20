import cv2
import numpy as np
import matplotlib.pyplot as plt

"""
1. Обрати бібліотеку із вже готовими функціями для лінійних трансформацій
(Рекомендую OpenCV, вона має найширший інструментарій);

2. За допомогою інструментарію бібліотеки реалізувати 
всі лінійни трансформації з пункту 2 попередньої частини
 на тих самих фігурах, створених у двовимірному просторі. 
 Порівняти результати, отримані за допомогою готових бібліотек, 
 з результатами роботи власних функцій
"""
def plot_cv_image(image, title="Image"):
    """ Convert BGR image to RGB and plot """
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def create_canvas(objects, shape=(500, 500, 3)):
    """ Creates a blank canvas and draws the list of objects """
    canvas = np.zeros(shape, dtype=np.uint8)
    for obj, color in objects:
        pts = np.array(obj, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=3)
    return canvas

# Define the objects on a larger scale for better visibility
object1 = np.array([[50, 50], [150, 250], [250, 350], [150, 450], [50, 350], [50, 50]]) * 1.5
object2 = np.array([[50, 50], [125, 250], [250, 125], [375, 250], [500, 50], [250, -125], [125, -250], [50, -125], [50, 50]]) * 0.75

canvas = create_canvas([(object1, (255, 0, 0)), (object2, (0, 255, 0))])
plot_cv_image(canvas, "Original Objects")


def apply_transformation(canvas, matrix):
    rows, cols = canvas.shape[:2]
    transformed_image = cv2.warpAffine(canvas, matrix, (cols, rows))
    return transformed_image

# Rotation
angle = 45
rotation_matrix = cv2.getRotationMatrix2D((250, 250), angle, 1)
rotated_image = apply_transformation(canvas, rotation_matrix)
plot_cv_image(rotated_image, "Rotated by 45 degrees using OpenCV")

# Scaling
scaling_matrix = np.float32([[2, 0, 0], [0, 0.5, 0]])  # Scale x by 2, y by 0.5
scaled_image = apply_transformation(canvas, scaling_matrix)
plot_cv_image(scaled_image, "Scaled Image using OpenCV")

# Reflection
reflection_matrix = np.float32([[1, 0, 0], [0, -1, canvas.shape[0]]])
reflected_image = apply_transformation(canvas, reflection_matrix)
plot_cv_image(reflected_image, "Reflected Image along X-axis")

"""
3. Взяти довільне зображення, зчитати його за допомогою 
image = cv2.imread('image.jpg') та виконати 2-3 лінійні трансформації 
над цим зображенням. Вивести результуючі зображення (1 бал).
"""

image_path = '/Users/klishchyuriy/Desktop/Applied_Linear_Algebra/lab_1/math_guy.jpeg'
image = cv2.imread(image_path)
if image is not None:
    plot_cv_image(image, "Original Image")
    rotated_real_image = apply_transformation(image, rotation_matrix)
    sheared_real_image = apply_transformation(image, scaling_matrix)
    reflected_real_image = apply_transformation(image, reflection_matrix)

    plot_cv_image(rotated_real_image, "Image Rotated by 45 Degrees")
    plot_cv_image(sheared_real_image, "Sheared Real Image")
    plot_cv_image(reflected_real_image, "Reflected Real Image along X-axis")
else:
    print("Failed to load image. Check the file path.")