# Implement matrix multiplication with @ operator in Python

class matrix(list[list]):
    def __matmul__(self, B):
        A = self
        return matrix([[sum(A[i][k]*B[k][j] for k in range(len(B)))
                    for j in range(len(B[0])) ] for i in range(len(A))])