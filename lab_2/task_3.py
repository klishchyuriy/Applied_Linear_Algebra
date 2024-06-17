import numpy as np

def encrypt_message(message, key_matrix):
    message_vector = np.array([ord(char) for char in message])
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    encrypted_vector = np.dot(diagonalized_key_matrix, message_vector)
    return encrypted_vector

def decrypt_message(encrypted_vector, key_matrix):
    eigenvalues, eigenvectors = np.linalg.eig(key_matrix)
    diagonalized_key_matrix = np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.linalg.inv(eigenvectors))
    decrypted_vector = np.dot(np.linalg.inv(diagonalized_key_matrix), encrypted_vector)
    decrypted_message = ''.join([chr(int(round(num.real))) for num in decrypted_vector])
    return decrypted_message

message = "Hello, World!"
print("Original Message:", message)
key_matrix = np.random.randint(0, 256, (len(message), len(message)))
encrypted_vector = encrypt_message(message, key_matrix)
print("Encrypted Message:\n", encrypted_vector)
decrypted_message = decrypt_message(encrypted_vector, key_matrix)
print("Decrypted Message:", decrypted_message)
