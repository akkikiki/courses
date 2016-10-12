# http://matplotlib.org/users/pyplot_tutorial.html
import numpy as np
import matplotlib.pyplot as plt
#ylim=(0.70, 1.00)
#if ylim is not None:
#    plt.ylim(*ylim)
a = np.arange(20)
plt.plot([1,2], [1,4], 'ro')
plt.plot([4], [3], 'bo')
#plt.plot(a, 3.0/2*a)
plt.plot(a, 4.0/5*a)
plt.axis([0, 5, 0, 5])
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.show()

