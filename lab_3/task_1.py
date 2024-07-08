import numpy as np

def my_svd(matrix):
    eigenvalues_matrix, eigenvectors_matrix = np.linalg.eig(np.dot(matrix, matrix.T))
    sorted_indices_matrix = np.argsort(eigenvalues_matrix)[::-1]
    U = eigenvectors_matrix[:, sorted_indices_matrix]

    eigenvalues_matrix_T, eigenvectors_matrix_T = np.linalg.eig(np.dot(matrix.T, matrix))
    sorted_indices_matrix_T = np.argsort(eigenvalues_matrix_T)[::-1]
    V = eigenvectors_matrix_T[:, sorted_indices_matrix_T]

    singular_values = np.sqrt(np.maximum(eigenvalues_matrix_T[sorted_indices_matrix_T], 0))
    Σ = np.zeros(matrix.shape)
    Σ[:min(matrix.shape), :min(matrix.shape)] = np.diag(singular_values)

    for i in range(len(singular_values)):
        if singular_values[i] != 0:
            U[:, i] = np.dot(matrix, V[:, i]) / singular_values[i]
        else:
            U[:, i] = np.zeros(matrix.shape[0])

    return U, Σ, V.T

# Test with a sample matrix
matrix = np.array([[0, 5], [1, 3], [1, 3]])
U, Σ, Vt = my_svd(matrix)

print("U: \n", U)
print("Σ: \n", Σ)
print("V^T: \n", Vt)
print("Reconstructed matrix: \n", np.dot(U, np.dot(Σ, Vt)).round(1))
print("Original matrix: \n", matrix)
