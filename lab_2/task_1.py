import numpy as np

def compute_eigenvalues_and_vectors(matrix):
    values, vectors = np.linalg.eig(matrix)
    return values, vectors

def validate_eigen(matrix, values, vectors):
    for i in range(len(values)):
        vector = vectors[:, i]
        lambda_vector = values[i] * vector
        matrix_vector = np.dot(matrix, vector)
        if not np.allclose(matrix_vector, lambda_vector):
            return False
    return True

matrix = np.array([[7, -2], [1, -1]])

values, vectors = compute_eigenvalues_and_vectors(matrix)

print("Eigenvalues:", values)
print("Eigenvectors:\n", vectors)

is_valid = validate_eigen(matrix, values, vectors)
print(is_valid)
