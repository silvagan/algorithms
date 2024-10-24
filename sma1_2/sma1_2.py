import numpy as np
from sympy import Abs, symbols, diff, cos, sin, lambdify, true, factorial
import matplotlib.pyplot as plt
import math
#2 dalis

x = symbols('x')
f = 5 * cos(x) - cos(2 * x) + 2

def derivative_func(func, degree):
    x = symbols('x')
    derivative = func
    for _ in range(degree):
        derivative = diff(derivative, x)
    return derivative

def lambdify_func(f, value):
    x = symbols('x')
    derivative_func = lambdify(x, f)
    return derivative_func(value)

dx= 0.1
x=np.arange(-6, 3+dx, dx)
y = lambdify_func(f, x)
plt.plot(x, y, 'b')
plt.grid()

def ts(x, x0, elementCount):
  if (elementCount > 0):
    return (np.power((x-x0), elementCount) / math.factorial(elementCount)) * lambdify_func(derivative_func(f, elementCount), x0) + ts(x, x0, elementCount - 1)
  else:
    return lambdify_func(f, x0)

x0 = -1.5; # given point

# values of Taylor Series
ts3 = ts(x, x0, 3)
ts4 = ts(x, x0, 4)
ts5 = ts(x, x0, 5)
tsy = ts(x, x0, 26)
#print(tsy)

plt.plot(x, y, 'r',  linewidth=3.0) # representing given function

# plt.plot(x, ts3, 'y', linewidth=3.0) # representing given function
# plt.plot(x, ts4, 'g', linewidth=4) # representing given function
# plt.plot(x, ts5, 'b', linewidth=2.0) # representing given function

plt.plot(x, tsy, 'k', 10) # representing given function
plt.plot(x0, lambdify_func(f, x0), 'ok') # representing reference point

plt.grid(True)

plt.xlim([-8, 6]); plt.ylim([-8, 10])

# xmin = -6
# xmax = 3
# h = 0.1
# tolerance = 1e-12
# r1=1
# r2=1
# while xmin<xmax:
#   kx1_1 = xmin
#   kx1_2 = xmin+h
#   kx2_1 = xmin
#   kx2_2 = xmin+h

#   kxmid1 = 1
#   kxmid2 = 1

#   a = 14
#   f1=lambdify_func(f, kx1_1); f2=lambdify_func(f, kx1_2);
#   h1=ts(kx2_1, x0, a); h2=ts(kx2_2, x0, a);

#   e1=False;e2=False
#   if np.sign(lambdify_func(f, xmin)) != np.sign(lambdify_func(f, xmin + h)):
#     e1=True
#     i=0
#     while np.abs(lambdify_func(f, kxmid1)) > tolerance:
#     #quasi-newton method
#       kxmid1 = kx1_1-(kx1_1-kx1_2)/(f1-f2)*f1;
#       fmid=lambdify_func(f, kxmid1); 
#       kx1_2=kx1_1; f2=f1; kx1_1=kxmid1; f1=fmid;   
#       i += 1
#     print("\n ========root [{}]========  [%.3f; %.3f]".format(r1) % (xmin,xmin+h))
#     print("\n[cross section]")
#     print("   result : %.20f" % (kxmid1))
#     print("   iterations : %.0f" % (i))
#     r1+=1

#   if np.sign(ts(xmin, x0, a)) != np.sign(ts(xmin + h, x0, a)):
#     e2=True
#     i=0
#     while np.abs(ts(kxmid2, x0, a)) > tolerance:
#     #quasi-newton method
#       kxmid2 = kx2_1-(kx2_1-kx2_2)/(h1-h2)*h1;
#       fmid=ts(kxmid2, x0, a); 
#       kx2_2=kx2_1; h2=h1; kx2_1=kxmid2; h1=fmid;   
#       i += 1
#     print("\n ========root [{}] (TE)========  [%.3f; %.3f]".format(r2) % (xmin,xmin+h))
#     print("\n[cross section]")
#     print("   result : %.20f" % (kxmid2))
#     print("   iterations : %.0f" % (i))
#     r2+=1
#   if e1 and e2:
#     print("   deviation : %.20f" % (Abs(kxmid2/kxmid1-1)))
   
#   xmin += h

x = symbols('x')
endFunction = f
derivative = diff(f)
for i in range(1,26):
 endFunction += (((x-x0)**i) / factorial(i)) * derivative
 derivative = diff(derivative)
print(endFunction)

plt.show()