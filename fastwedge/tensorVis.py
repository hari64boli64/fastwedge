import itertools
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


def __vis_subroutine(N: int,
                     twoD: List[List[complex]],
                     size: int,
                     title: Union[str, None] = None) -> dict:
    fig = plt.figure(figsize=(size, size), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(twoD, origin="upper")
    plt.colorbar()
    plt.title(title if title is not None else f"N:{N}", y=1.2, fontsize=15)
    params = {"va": "center", "ha": "center",
              "fontsize": 12, "transform": ax.transAxes}
    return params


def __vis2dTensorKarnaugh(tensor: np.ndarray, size: int,
                          title: Union[str, None]):
    """2d tensor visualizer

    Args:
        tensor (np.ndarray): target tensor
        size (int): size of the output img
        title (str): title of the fig
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 2, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**2 <= 10000000, f"N^2={N**2}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j])
         for j in range(N)]
        for i in range(N)
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text(-0.05, 1-(x+1/2)/(N), str(i), **params)
        x += 1
    plt.text(-0.05, 1.02, "i", color="blue", **params)
    y = 0
    for j in range(N):
        plt.text((y+1/2)/(N), 1.05, str(j),  **params)
        y += 1
    plt.text(-0.02, 1.05, "j",  color="blue", **params)
    plt.show()


def __vis4dTensorKarnaugh(tensor: np.ndarray, size: int,
                          title: Union[str, None]):
    """4d tensor visualizer(Karnaugh)

    Args:
        tensor (np.ndarray): target tensor
        size (int): size of the output img
        title (str): title of the fig
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 4, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**4 <= 10000000, f"N^4={N**4}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l_])
         for k, l_ in itertools.product(range(N), range(N))]
        for i, j in itertools.product(range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text(-0.1, 1-(x+N/2)/(N*N), str(i), **params)
        for j in range(N):
            plt.text(-0.05, 1-(x+1/2)/(N*N), str(j),  **params)
            x += 1
    plt.text(-0.10, 1.02,  "i", color="blue", **params)
    plt.text(-0.05, 1.02, "j", color="blue", **params)
    y = 0
    for k in range(N):
        plt.text((y+N/2)/(N*N), 1.10, str(k),  **params)
        for l_ in range(N):
            plt.text((y+1/2)/(N*N), 1.05, str(l_), **params)
            y += 1
    plt.text(-0.02, 1.10,  "k",  color="blue", **params)
    plt.text(-0.02, 1.05, "l",  color="blue", **params)
    plt.show()


def __vis6dTensorKarnaugh(tensor: np.ndarray, size: int,
                          title: Union[str, None]):
    """6d tensor visualizer

    Args:
        tensor (np.ndarray): target tensor
        size (int): size of the output img
        title (str): title of the fig
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 6, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**6 <= 10000000, f"N^6={N**6}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l_][m][n])
         for l_, m, n in itertools.product(range(N), range(N), range(N))]
        for i, j, k in itertools.product(range(N), range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text(-0.15, 1-(x+N*N/2)/(N*N*N), str(i), **params)
        for j in range(N):
            plt.text(-0.10, 1-(x+N/2)/(N*N*N), str(j),  **params)
            for k in range(N):
                plt.text(-0.05, 1-(x+1/2)/(N*N*N), str(k),  **params)
                x += 1
    plt.text(-0.15, 1.02, "i", color="blue", **params)
    plt.text(-0.10, 1.02, "j", color="blue", **params)
    plt.text(-0.05, 1.02, "k", color="blue", **params)
    y = 0
    for l_ in range(N):
        plt.text((y+N*N/2)/(N*N*N), 1.15, str(l_),  **params)
        for m in range(N):
            plt.text((y+N/2)/(N*N*N), 1.10, str(m), **params)
            for n in range(N):
                plt.text((y+1/2)/(N*N*N), 1.05, str(n), **params)
                y += 1
    plt.text(-0.02, 1.15, "l",  color="blue", **params)
    plt.text(-0.02, 1.10, "m",  color="blue", **params)
    plt.text(-0.02, 1.05, "n",  color="blue", **params)
    plt.show()


def __vis8dTensorKarnaugh(tensor: np.ndarray, size: int,
                          title: Union[str, None]):
    """8d tensor visualizer

    Args:
        tensor (np.ndarray): target tensor
        size (int): size of the output img
        title (str): title of the fig
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 8, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**8 <= 100000000, f"N^8={N**8}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l_][m][n][o][p])
         for m, n, o, p in itertools.product(range(N), range(N),
                                             range(N), range(N))]
        for i, j, k, l_ in itertools.product(range(N), range(N),
                                             range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text(-0.20, 1-(x+N*N*N/2)/(N*N*N*N),  str(i), **params)
        for j in range(N):
            plt.text(-0.15, 1-(x+N*N/2)/(N*N*N*N),  str(j),  **params)
            for k in range(N):
                plt.text(-0.10, 1-(x+N/2)/(N*N*N*N),  str(k),  **params)
                for l_ in range(N):
                    plt.text(-0.05, 1-(x+1/2)/(N*N*N*N),  str(l_),  **params)
                    x += 1
    plt.text(-0.20, 1.02, "i", color="blue", **params)
    plt.text(-0.15, 1.02, "j", color="blue", **params)
    plt.text(-0.10, 1.02, "k", color="blue", **params)
    plt.text(-0.05, 1.02, "l", color="blue", **params)
    y = 0
    for m in range(N):
        plt.text((y+N*N*N/2)/(N*N*N*N), 1.20, str(m),  **params)
        for n in range(N):
            plt.text((y+N*N/2)/(N*N*N*N), 1.15, str(n), **params)
            for o in range(N):
                plt.text((y+N/2)/(N*N*N*N), 1.10, str(o), **params)
                for p in range(N):
                    plt.text((y+1/2)/(N*N*N*N), 1.05,  str(p), **params)
                    y += 1
    plt.text(-0.02, 1.20, "m",  color="blue", **params)
    plt.text(-0.02, 1.15, "n",  color="blue", **params)
    plt.text(-0.02, 1.10, "o",  color="blue", **params)
    plt.text(-0.02, 1.05, "p",  color="blue", **params)
    plt.show()


def visTensor(tensor: np.ndarray, size: int = 8,
              title: Union[str, None] = None):
    """tensor visualizer

    Args:
        tensor (np.ndarray): target tensor
        size (int, optional): size of the output img. Defaults to 8.
        title (Union[str, None], optional): title of the fig. Defaults to None.
    """
    if tensor.ndim == 2:
        __vis2dTensorKarnaugh(tensor, size, title)
    elif tensor.ndim == 4:
        __vis4dTensorKarnaugh(tensor, size, title)
    elif tensor.ndim == 6:
        __vis6dTensorKarnaugh(tensor, size, title)
    elif tensor.ndim == 8:
        __vis8dTensorKarnaugh(tensor, size, title)
    else:
        assert False, f"tensor.ndim must be 2,4,6, or 8, but is {tensor.ndim}"


def vis4dTensorNest(tensor: np.ndarray, size: int = 8,
                    title: Union[str, None] = None):
    """4d tensor visualizer(Nest)

    Args:
        tensor (np.ndarray): target tensor
        size (int, optional): size of the output img. Defaults to 8.
        title (Union[str, None], optional): title of the fig. Defaults to None.
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 4, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**4 <= 10000000, f"N^4={N**4}, which is too big to vis."
    listTensor = tensor.tolist()
    twoD = [
        [abs(listTensor[i][j][k][l])
         for j, l in itertools.product(range(N), range(N))]
        for i, k in itertools.product(range(N), range(N))
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text(-0.10, 1-(x+N/2)/(N*N), str(i), **params)
        for k in range(N):
            plt.text(-0.05, 1-(x+1/2)/(N*N),  str(k),  **params)
            x += 1
    plt.text(-0.10, 1.02,  "i", color="blue", **params)
    plt.text(-0.05, 1.02, "k", color="blue", **params)
    y = 0
    for j in range(N):
        plt.text((y+N/2)/(N*N), 1.10, str(j),  **params)
        for l_ in range(N):
            plt.text((y+1/2)/(N*N), 1.05, str(l_), **params)
            y += 1
    plt.text(-0.02, 1.10, "j",  color="blue", **params)
    plt.text(-0.02, 1.05, "l",  color="blue", **params)
    plt.show()


def vis2dTensorReal(tensor: np.ndarray, size: int = 8,
                    title: Union[str, None] = None):
    """2d tensor visualizer of real

    Args:
        tensor (np.ndarray): target real tensor
        size (int, optional): size of the output img. Defaults to 8.
        title (Union[str, None], optional): title of the fig. Defaults to None.
    """
    N = tensor.shape[0]
    assert len(tensor.shape) == 2, tensor.shape
    assert len(set(tensor.shape)) == 1, tensor.shape
    assert N**2 <= 10000000, f"N^2={N**2}, which is too big to vis."
    assert np.all(tensor.imag == 0.0)
    listTensor = tensor.tolist()
    twoD = [
        [listTensor[i][j].real
         for j in range(N)]
        for i in range(N)
    ]
    params = __vis_subroutine(N, twoD, size, title)
    x = 0
    for i in range(N):
        plt.text(-0.05, 1-(x+1/2)/(N), str(i), **params)
        x += 1
    plt.text(-0.05, 1.02, "i", color="blue", **params)
    y = 0
    for j in range(N):
        plt.text((y+1/2)/(N), 1.05, str(j),  **params)
        y += 1
    plt.text(-0.02, 1.05, "j",  color="blue", **params)
    plt.show()
