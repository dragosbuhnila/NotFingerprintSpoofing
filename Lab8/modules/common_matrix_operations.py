import numpy as np


def _matrix_is_colvector(matrix):
    if matrix.ndim == 2 and matrix.shape[1] == 1:
        return True
    else:
        return False


def _matrix_is_rowvector(matrix):
    if matrix.ndim == 2 and matrix.shape[0] == 1:
        return True
    else:
        return False


def _matrix_is_1d(matrix):
    if matrix.ndim == 1:
        return True
    else:
        return False


def _matrix_is_convertible_to_1d(matrix):
    if not (_matrix_is_1d(matrix) or _matrix_is_colvector(matrix) or _matrix_is_rowvector(matrix)):
        return False
    else:
        return True


'''
Should only use on (n, ) (1,n), numpy arrays (or already column vectors, i.e. (n, 1))
'''
def vcol(matrix):
    if np.isscalar(matrix):
        column_vector = np.array(matrix)
        column_vector = column_vector.reshape((1, 1))
        return column_vector

    if not _matrix_is_convertible_to_1d(matrix):
        raise ValueError("Input must be either a (1,n) or a (n,) numpy array")
    column_vector = matrix.reshape((matrix.size, 1))
    return column_vector


'''
Should only use on (n, ) (n, 1), numpy arrays (or already row vectors, i.e. (1, n))
'''
def vrow(matrix):
    if np.isscalar(matrix):
        column_vector = np.array(matrix)
        column_vector = column_vector.reshape((1, 1))
        return column_vector

    if not _matrix_is_convertible_to_1d(matrix):
        raise ValueError("Input must be either a (n,1) or a (n,) numpy array")
    row_vector = matrix.reshape((1, matrix.size))
    return row_vector


def to_numpy_matrix(x: np.array, vector_type: str):
    if np.isscalar(x):
        x = np.array(x)

    if _matrix_is_1d(x):
        if vector_type == "col":
            x = vcol(x)
        elif vector_type == "row":
            x = vrow(x)
        else:
            raise TypeError("In 'to_numpy_matrix(x, vector_type)' vector_type isn't one of ['col', row']")

    return x


if __name__ == "__main__":
    test_matrix = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]).reshape(3, 3)
    # test_matrix = np.array([1, 2, 3]).reshape(1, 3)
    # test_matrix = np.array([1, 2, 3]).reshape(3, 1)
    # test_matrix = np.array([1, 2, 3])
    print("test_matrix is:")
    print(test_matrix)
    print()

    # print(f"Is matrix 1d? {_matrix_is_1d(test_matrix)}")
    # print(f"Is matrix convertible to 1d? {_matrix_is_convertible_to_1d(test_matrix)}")

    print(vcol(test_matrix))

