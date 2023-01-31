import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


def __vis_subroutine(N: int,
                     twoD: List[List[complex]],
                     size: int,
                     title: Union[str, None] = None) -> dict:
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(twoD, origin="upper")
    plt.colorbar()
    plt.title(title if title is not None else f"N:{N}", y=1.2, fontsize=15)
    params = {"va": "center", "ha": "center",
              "fontsize": 12, "transform": ax.transAxes}
    return params


def vis2dTensorKarnaugh(tensor: np.ndarray, size: int = 8, title=None):
    """2d tensor visualizer

    Args:
        tensor (np.ndarray): target tensor
        size (int, optional): size of output img. Defaults to 8.
        title (_type_, optional): title of the fig. Defaults to None.
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 2, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**2 <= 10000000, f"N^2={N**2}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j])
         for i in range(N)]
        for j in range(N)
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text((x+1/2)/(N), 1.05, str(i), **params)
        x += 1
    plt.text(-0.02, 1.05, "i", color="blue", **params)
    y = 0
    for j in range(N):
        plt.text(-0.05, 1-(y+1/2)/(N), str(j),  **params)
        y += 1
    plt.text(-0.05, 1.02, "j",  color="blue", **params)
    plt.show()


def vis4dTensorKarnaugh(tensor: np.ndarray, size: int = 8, title=None):
    """4d tensor visualizer(Karnaugh)

    Args:
        tensor (np.ndarray): target tensor
        size (int, optional): size of output img. Defaults to 8.
        title (_type_, optional): title of the fig. Defaults to None.
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 4, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**4 <= 10000000, f"N^4={N**4}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l])
         for i, j in itertools.product(range(N), range(N))]
        for k, l in itertools.product(range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text((x+N/2)/(N*N), 1.1, str(i), **params)
        for j in range(N):
            plt.text((x+1/2)/(N*N), 1.05, str(j),  **params)
            x += 1
    plt.text(-0.02, 1.1, "i", color="blue", **params)
    plt.text(-0.02, 1.05, "j", color="blue", **params)
    y = 0
    for k in range(N):
        plt.text(-0.1, 1-(y+N/2)/(N*N), str(k),  **params)
        for l in range(N):
            plt.text(-0.05, 1-(y+1/2)/(N*N), str(l), **params)
            y += 1
    plt.text(-0.1, 1.02, "k",  color="blue", **params)
    plt.text(-0.05, 1.02, "l",  color="blue", **params)
    plt.show()


def vis4dTensorNest(tensor: np.ndarray, size: int = 8, title=None):
    """4d tensor visualizer(Nest)

    Args:
        tensor (np.ndarray): target tensor
        size (int, optional): size of output img. Defaults to 8.
        title (_type_, optional): title of the fig. Defaults to None.
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 4, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**4 <= 10000000, f"N^4={N**4}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l])
         for i, k in itertools.product(range(N), range(N))]
        for j, l in itertools.product(range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text((x+N/2)/(N*N), 1.1, str(i), **params)
        for k in range(N):
            plt.text((x+1/2)/(N*N), 1.05, str(k),  **params)
            x += 1
    plt.text(-0.02, 1.1, "i", color="blue", **params)
    plt.text(-0.02, 1.05, "k", color="blue", **params)
    y = 0
    for j in range(N):
        plt.text(-0.1, 1-(y+N/2)/(N*N), str(j),  **params)
        for l in range(N):
            plt.text(-0.05, 1-(y+1/2)/(N*N), str(l), **params)
            y += 1
    plt.text(-0.1, 1.02, "j",  color="blue", **params)
    plt.text(-0.05, 1.02, "l",  color="blue", **params)
    plt.show()


def vis6dTensorKarnaugh(tensor: np.ndarray, size: int = 8, title=None):
    """6d tensor visualizer

    Args:
        tensor (np.ndarray): target tensor
        size (int, optional): size of output img. Defaults to 8.
        title (_type_, optional): title of the fig. Defaults to None.
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 6, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**6 <= 10000000, f"N^6={N**6}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l][m][n])
         for i, j, k in itertools.product(range(N), range(N), range(N))]
        for l, m, n in itertools.product(range(N), range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text((x+N*N/2)/(N*N*N), 1.15, str(i), **params)
        for j in range(N):
            plt.text((x+N/2)/(N*N*N), 1.10, str(j),  **params)
            for k in range(N):
                plt.text((x+1/2)/(N*N*N), 1.05, str(k),  **params)
                x += 1
    plt.text(-0.02, 1.15, "i", color="blue", **params)
    plt.text(-0.02, 1.10, "j", color="blue", **params)
    plt.text(-0.02, 1.05, "k", color="blue", **params)
    y = 0
    for l in range(N):
        plt.text(-0.15, 1-(y+N*N/2)/(N*N*N), str(l),  **params)
        for m in range(N):
            plt.text(-0.10, 1-(y+N/2)/(N*N*N), str(m), **params)
            for n in range(N):
                plt.text(-0.05, 1-(y+1/2)/(N*N*N), str(n), **params)
                y += 1
    plt.text(-0.15, 1.02, "l",  color="blue", **params)
    plt.text(-0.10, 1.02, "m",  color="blue", **params)
    plt.text(-0.05, 1.02, "n",  color="blue", **params)
    plt.show()
