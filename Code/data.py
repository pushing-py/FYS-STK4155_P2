import numpy as np


def generate_dataset(n=1000, noise_std=1.0, random_state=42):
    """
    Genererer et datasett basert på funksjonen f(x) = 1 / (1 + 25x^2)
    med normalfordelt støy.
    
    Parametre
    ---------
    n : int
        Antall datapunkter (default 500).
    noise_std : float
        Standardavvik for støyen (default 1.0).
    random_state : int
        Seed for random number generator (default 42).
    
    Returnerer
    ----------
    x : np.ndarray
        x-verdier.
    y : np.ndarray
        "Sann" funksjonsverdi uten støy.
    y_noisy : np.ndarray
        Verdier med støy lagt til.
    """
    np.random.seed(random_state)
    x = np.linspace(-1, 1, n)
    y = 1 / (1 + 25 * x**2)
    y_noisy = y + np.random.normal(0, noise_std, n)
    return x, y, y_noisy
