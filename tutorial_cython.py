import pandas as pd
import numpy as np
import cython_functions

df = pd.DataFrame({'a': np.random.randn(10000),
                   'b': np.random.randn(10000),
                   'N': np.random.randint(100, 1000, 10000),
                   'x': 'x'})

def f(x):
    return x * (x - 1)


def integrate_f(a, b, N):
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f(a + i * dx)
    return s * dx


def compute_cython(df):
    result = cython_functions.apply_integrate_f(df['a'].to_numpy(),
                                   df['b'].to_numpy(),
                                   df['N'].to_numpy())
    return pd.Series(result, index=df.index, name='result')

df = compute_cython(df)
_ = 1