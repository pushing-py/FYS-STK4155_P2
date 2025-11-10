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

def mse(y, y_hat):
    return np.mean((y - y_hat)**2)

def sgd_FFNN(X_train, y_train, X_test, y_test, activation, dactivation, layers=[1,10,1], eta=0.01, num_steps=1000,
             beta1=0.9, beta2=0.999, eps=1e-8, batch_size=32, seed=0, optimizer="adam"):
    
    rng = np.random.default_rng(seed)
    n, d = X_train.shape
    # --- initialize weights ---
    W, b = [], []
    for i in range(len(layers)-1):
        W.append(rng.normal(0, 1/np.sqrt(layers[i]), (layers[i], layers[i+1])))
        b.append(np.zeros((1, layers[i+1])))
    # --- initialize Adam params ---
    mW = [np.zeros_like(w) for w in W]
    vW = [np.zeros_like(w) for w in W]
    mb = [np.zeros_like(bb) for bb in b]
    vb = [np.zeros_like(bb) for bb in b]

    mse_train, mse_test = [], []
    
    for t in range(1, num_steps+1):
        idx = rng.choice(n, batch_size, replace=False)
        Xb, yb = X_train[idx], y_train[idx]
        
        # --- forward ---
        A = [Xb]
        for i in range(len(W)-1):
            Z = A[-1] @ W[i] + b[i]
            A.append(activation(Z))
        ZL = A[-1] @ W[-1] + b[-1]   # linear output
        yhat = ZL

        # --- backward ---
        dZ = 2.0*(yhat - yb)/batch_size
        dW, db = [None]*len(W), [None]*len(W)
        dW[-1] = A[-1].T @ dZ
        db[-1] = np.sum(dZ, axis=0, keepdims=True)
        dA = dZ @ W[-1].T
        for i in range(len(W)-2, -1, -1):
            dZ = dA * dactivation(A[i+1])
            dW[i] = A[i].T @ dZ
            db[i] = np.sum(dZ, axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ W[i].T
       # --- parameter update ---
        for i in range(len(W)):

            if optimizer.lower() == "sgd":
            # vanilla SGD
                W[i] -= eta * dW[i]
                b[i] -= eta * db[i]

            elif optimizer.lower() == "rmsprop":
            # RMSProp
                vW[i] = beta2*vW[i] + (1-beta2)*(dW[i]**2)
                vb[i] = beta2*vb[i] + (1-beta2)*(db[i]**2)
                W[i] -= eta * dW[i] / (np.sqrt(vW[i]) + eps)
                b[i] -= eta * db[i] / (np.sqrt(vb[i]) + eps)

            elif optimizer.lower() == "adam":
        # Adam
                mW[i] = beta1*mW[i] + (1-beta1)*dW[i]
                vW[i] = beta2*vW[i] + (1-beta2)*(dW[i]**2)
                mb[i] = beta1*mb[i] + (1-beta1)*db[i]
                vb[i] = beta2*vb[i] + (1-beta2)*(db[i]**2)
                mW_hat = mW[i] / (1 - beta1**t)
                vW_hat = vW[i] / (1 - beta2**t)
                mb_hat = mb[i] / (1 - beta1**t)
                vb_hat = vb[i] / (1 - beta2**t)
                W[i] -= eta * mW_hat / (np.sqrt(vW_hat) + eps)
                b[i] -= eta * mb_hat / (np.sqrt(vb_hat) + eps)
            else:
                raise ValueError("optimizer must be 'sgd', 'rmsprop' or 'adam'")

        def forward(X):
            A = X
            for i in range(len(W)-1):
                A = activation(A @ W[i] + b[i])
            return A @ W[-1] + b[-1]

        mse_train.append(mse(y_train, forward(X_train)))
        mse_test.append(mse(y_test, forward(X_test)))
    return W, b, mse_train, mse_test

def ReLU(z):
    return np.where(z > 0, z, 0)

def dReLU(a):
    return np.where(a > 0, 1, 0)

def leaky_ReLU(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def dleaky_ReLU(a, alpha=0.01):
    return np.where(a > 0, 1, alpha)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def dsigmoid(a):
    return a * (1 - a)