import numpy as np

def strassen_matrix_multiply(A, B):
    n = len(A)

    if n == 1:
        return np.array([[A[0][0] * B[0][0]]])

    half_size = n // 2

    A11 = A[:half_size, :half_size]
    A12 = A[:half_size, half_size:]
    A21 = A[half_size:, :half_size]
    A22 = A[half_size:, half_size:]

    B11 = B[:half_size, :half_size]
    B12 = B[:half_size, half_size:]
    B21 = B[half_size:, :half_size]
    B22 = B[half_size:, half_size:]

    P1 = strassen_matrix_multiply(A11, B12 - B22)
    P2 = strassen_matrix_multiply(A11 + A12, B22)
    P3 = strassen_matrix_multiply(A21 + A22, B11)
    P4 = strassen_matrix_multiply(A22, B21 - B11)
    P5 = strassen_matrix_multiply(A11 + A22, B11 + B22)
    P6 = strassen_matrix_multiply(A12 - A22, B21 + B22)
    P7 = strassen_matrix_multiply(A11 - A21, B11 + B12)

    C11 = P5 + P4 - P2 + P6
    C12 = P1 + P2
    C21 = P3 + P4
    C22 = P5 + P1 - P3 - P7

    result = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))

    return result

n = 2
A = np.random.randint(1, 10, (n, n))
B = np.random.randint(1, 10, (n, n))

print("Matrix A:")
print(A)
print("\nMatrix B:")
print(B)

result = strassen_matrix_multiply(A, B)
print("\nResult Matrix C:")
print(result)
