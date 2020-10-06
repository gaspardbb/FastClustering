import numpy as np
import clustering as m

a = np.arange(1, 4, dtype='float32')
a = a / a.sum()

sampler_f = m.VoseAlias(a)
sampler_d = m.VoseAlias(a.astype('float64'))

sampler_f.sample(5)
sampler_d.sample(10)

matrix = np.zeros((3, 2), order='C', dtype='float64')

m.kmeanspp(matrix, 2, 0.)
m.kmeanspp(matrix, 2, 10)

x_1 = np.random.random_sample((5, 2))
x_2 = np.random.random_sample((4, 2))
res = m.entropic_wasserstein_sin(x_1, x_2, .1)
print(f"Error message is: {res.error_msg}")
print(f"Duality gap before changing: {res.duality_gap}")

from clustering.utils.plot_utils import add_linear_regression

import matplotlib.pyplot as plt
plt.plot(res.errors)
plt.xscale('log')
plt.yscale('log')
plt.show()