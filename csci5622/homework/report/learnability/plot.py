# http://matplotlib.org/users/pyplot_tutorial.html
import numpy as np
import matplotlib.pyplot as plt
#ylim=(0.70, 1.00)
#if ylim is not None:
#    plt.ylim(*ylim)
a = np.arange(20)
## Pattern 1
#plt.plot([2], [4], 'ro', ms=12)
#plt.plot([3], [3], 'bo', ms=12)
#plt.plot(a, 4.0/5*a)


## Pattern 2
plt.plot([2], [4], 'bo', ms=12)
plt.plot([3], [3], 'ro', ms=12)
plt.plot(a, 5.0/2*a)

## Pattern 3
#plt.plot([2], [4], 'ro', ms=12)
#plt.plot([3], [3], 'ro', ms=12)
#plt.plot(a, 5.0/1*a)

## Pattern 4
#plt.plot([2], [4], 'bo', ms=12)
#plt.plot([3], [3], 'bo', ms=12)
#plt.plot(a, 4.0/3*a)



plt.axis([0, 5, 0, 5])
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.show()

