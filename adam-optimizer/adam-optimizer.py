import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    param, grad, m, v = (np.asarray(param, dtype=float), 
                        np.asarray(grad, dtype=float),
                        np.asarray(m, dtype=float), 
                        np.asarray(v, dtype=float))

    mt = beta1 * m + (1-beta1)*grad

    vt = beta2 * v + (1-beta2)*(grad**2)

    mhat = mt/(1-(beta1**t))
    vhat = vt/(1-(beta2**t))

    paramt = param - lr * (mhat/(np.sqrt(vhat)+eps))

    return paramt, mt, vt