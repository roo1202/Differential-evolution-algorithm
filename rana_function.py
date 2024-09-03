import numpy as np

def rana_function(x: np.ndarray) -> float:
    """
    Calcula el valor de la función rana_function para un vector dado x.

    Args:
        x (np.ndarray): Vector de dimensión D.

    Returns:
        float: Valor de la función rana_function evaluada en x.
    """
    D = len(x)
    total = 0.0
    for i in range(D - 1):
        xi = x[i]
        xi_plus_1 = x[i + 1]
        t1 = np.sqrt(np.abs(xi_plus_1 + xi + 1))
        t2 = np.sqrt(np.abs(xi_plus_1 - xi + 1))
        term1 = (xi_plus_1 + 1) * np.cos(t2) * np.sin(t1)
        term2 = xi * np.cos(t1) * np.sin(t2)
        total += term1 + term2
    return total
