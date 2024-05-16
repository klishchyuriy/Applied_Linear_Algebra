import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
"""
1. Для початку потрібно створити і візуалізувати об'єкти, 
на яких буде відображатись виконана лінійна трансформація у двовимірному просторі. 
Ваше завдання – створити два різні об'єкти. Це можуть бути як фігури, 
так і просто набір векторів (бажано уникати симетричних зображень, 
щоб трансформації були більш помітні)
"""
def plot_object(coords, title='Object'):
    plt.figure()
    plt.plot(coords[:, 0], coords[:, 1], 'o-')  # Plot as dots connected by lines
    plt.fill(coords[:, 0], coords[:, 1], alpha=0.3)  # Fill the area under the plot
    plt.gca().set_aspect('equal', adjustable='box')  # Equal scaling on both axes
    plt.title(title)
    plt.grid(True)
    plt.show()

# Define two different objects
object1 = np.array([[0, 0], [1, 2], [2, 3], [1, 4], [0, 3], [0, 0]])
object2 = np.array([[0, 0], [0.5, 1], [1, 0.5], [1.5, 1], [2, 0], [1, -0.5], [0.5, -1], [0, -0.5], [0, 0]])

# Visualize the objects
plot_object(object1, 'Object 1')
plot_object(object2, 'Object 2')


"""
2. Наступним кроком потрібно реалізувати функції, 
що будуть виконувати певні лінійні трансформації:

- обертати об'єкт на певний кут;
"""
def rotate_object(coords, angle_degrees):
    angle_radians = np.radians(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    print("Rotation Matrix:\n", rotation_matrix)
    transformed_coords = np.dot(coords, rotation_matrix)
    plot_object(transformed_coords, f'Rotated Object by {angle_degrees} degrees')
    return transformed_coords

# Test rotation
rotated_object = rotate_object(object1, 45)


"""
- маштабувати об'єкт з певним коефіцієнтом;
"""
def scale_object(coords, sx, sy):
    scaling_matrix = np.array([
        [sx, 0],
        [0, sy]
    ])
    print("Scaling Matrix:\n", scaling_matrix)
    transformed_coords = np.dot(coords, scaling_matrix)
    plot_object(transformed_coords, f'Scaled Object by factors {sx} and {sy}')
    return transformed_coords

# Test scaling
scaled_object = scale_object(object1, 2, 0.5)


"""
- віддзеркалювати об'єкт відносно певної осі;
"""
def reflect_object(coords, axis='x'):
    if axis == 'x':
        reflection_matrix = np.array([
            [1, 0],
            [0, -1]
        ])
    elif axis == 'y':
        reflection_matrix = np.array([
            [-1, 0],
            [0, 1]
        ])
    print("Reflection Matrix:\n", reflection_matrix)
    transformed_coords = np.dot(coords, reflection_matrix)
    plot_object(transformed_coords, f'Reflected Object across {axis.upper()}-axis')
    return transformed_coords

# Test reflection
reflect_object(object1, 'x')


"""
- робити нахил певної осі координат;
"""
def shear_object(coords, k, axis='x'):
    if axis == 'x':
        shear_matrix = np.array([
            [1, k],
            [0, 1]
        ])
    elif axis == 'y':
        shear_matrix = np.array([
            [1, 0],
            [k, 1]
        ])
    print("Shear Matrix:\n", shear_matrix)
    transformed_coords = np.dot(coords, shear_matrix)
    plot_object(transformed_coords, f'Sheared Object along {axis.upper()}-axis')
    return transformed_coords

# Test shear
shear_object(object1, 0.5, 'x')


"""
та універсальну функцію, що буде виконувати трансформацію 
з переданою у функцію кастомною матрицею трансформації.
"""
def custom_transform(coords, transformation_matrix):
    return np.dot(coords, transformation_matrix)

# Test custom transformation with an example matrix
example_matrix = np.array([
    [1, 1],
    [-1, 1]
])
custom_transformed_object = custom_transform(object1, example_matrix)
plot_object(custom_transformed_object, 'Custom Transformed Object 1')


"""
3. Поексперементувати з різними матрицями трансформації, 
зробити висновки, які елементи матриці на що впливають.

4. Спробувати виконати лінійні трансформації у тривимірному просторі, 
створивши принаймі одну тривиміну фігуру, 
та виконавши принаймі дві різні трансформації.
"""
def plot_object_3d(coords, title='3D Object'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], 'o-')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.title(title)
    plt.show()

def rotate_object_3d(coords, angle_degrees, axis='z'):
    angle_radians = np.radians(angle_degrees)
    if axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])
    # Include rotation matrices for x and y if needed
    transformed_coords = np.dot(coords, rotation_matrix)
    plot_object_3d(transformed_coords, f'3D Rotated Object around {axis.upper()} by {angle_degrees} degrees')

# Define a 3D object
object3d = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1], [0, 0, 0]])

# Visualize the original 3D object
plot_object_3d(object3d, 'Original 3D Object')

# Test 3D rotation
rotate_object_3d(object3d, 45)
