
import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sp

def fx(x): return -0.67 * (x ** 4) + 2.51 * (x ** 3) + 2.27 * (x ** 2) - 4.02 * x - 2.48
def gx(x): return (np.e ** (-x ** 2)) * (np.sin(x**2))*(x+2)
def hx(x): return 5 * math.cos(x) - math.cos(2 * x) + 12

Pi=math.pi
xx1=np.linspace(-4,5,100); #print(xxx)
xx2=np.linspace(-3,3,100);
fff=fx(xx1);
ggg=gx(xx2);

fig=plt.figure();fig.set_size_inches(10,5);ax1=fig.add_subplot(1,2,1);ax2=fig.add_subplot(1,2,2);
h = 0.1
tolerance = 1e-12

f1,=ax1.plot(xx1,fff,'b');ax1.grid();ax1.set_xlabel('x');ax1.set_ylabel('y');ax1.set_title('fx(x)');ax1.set_ylim(-30, 30)
xmin = -3.4495
xmax = 4.7463
print("\n.--------------------.")
print("|   function fx(x)   |")
print("'--------------------'")
print("scan interval: [{}; {}]".format(xmin, xmax))
print("deviation tolerance: {}".format(tolerance))

ri=0
while xmin<xmax:

  # bisection variables
  bx = xmin
  bx1 = xmin+h
  bxmid = 1
  bi = 0

  # kvazi-newton variables
  kx = xmin
  kx1 = xmin+h
  kxmid = 1
  f1=fx(kx); f2=fx(kx1);
  ki = 0


  if np.sign(fx(xmin)) !=np.sign(fx(xmin+h)):
    ri+=1
    while np.abs(fx(bxmid)) > tolerance:
        #bisection method (cut in half)
        bxmid = (bx+bx1)/2
        if np.sign(fx(bxmid)) == np.sign(fx(bx)):
            bx = bxmid
        else:
            bx1 = bxmid
        bi += 1

    while np.abs(fx(kxmid)) > tolerance:
        #quasi-newton method
        kxmid=kx-(kx-kx1)/(f1-f2)*f1;
        fmid=fx(kxmid); 
        kx1=kx; f2=f1; kx=kxmid; f1=fmid;   
        ki += 1
    
    print("\n ========root [{}]========  [%.3f; %.3f]".format(ri) % (xmin,xmin+h))
    print("[bisection]")
    print("   result : %.20f" % (bxmid))
    print("   iterations : %.0f" % (bi))
    print("\n[cross section]")
    print("   result : %.20f" % (kxmid))
    print("   iterations : %.0f" % (ki))
    
    ax1.plot([xmin], [0], 'or')
    ax1.plot([xmin+h], [0], 'or')

    # bisection result
    ax1.plot([bxmid], [0], 'ob')
    # quasi-newton(cross section) result
    ax1.plot([kxmid], [0], 'ok')

  xmin += h

f2,=ax2.plot(xx2,ggg,'r');ax2.grid();ax2.set_xlabel('x');ax2.set_ylabel('y');ax2.set_title('gx(x)');ax2.set_ylim(-2, 2)
xmin = -3
xmax = 3
print("\n\n.--------------------.")
print("|   function gx(x)   |")
print("'--------------------'")
print("scan interval: [{}; {}]".format(xmin, xmax))
print("deviation tolerance: {}".format(tolerance))
ri=0
while xmin<xmax:

  # bisection variables
  bx = xmin
  bx1 = xmin+h
  bxmid = 1
  bi = 0

  # kvazi-newton variables
  kx = xmin
  kx1 = xmin+h
  kxmid = 1
  g1=gx(kx); g2=gx(kx1);
  ki = 0

  tolerance = 1e-14


  if np.sign(gx(xmin)) !=np.sign(gx(xmin+h)):
    ri+=1
    while np.abs(gx(bxmid)) > tolerance:
        #bisection method (cut in half)
        bxmid = (bx+bx1)/2
        if np.sign(gx(bxmid)) == np.sign(gx(bx)):
            bx = bxmid
        else:
            bx1 = bxmid
        bi += 1

    while np.abs(gx(kxmid)) > tolerance:
        #quasi-newton method
        kxmid=kx-(kx-kx1)/(g1-g2)*g1;
        gmid=gx(kxmid); 
        kx1=kx; g2=g1; kx=kxmid; g1=gmid;   
        ki += 1
    
    print("\n ========root [{}]========  [%.3f; %.3f]".format(ri) % (xmin,xmin+h))
    print("[bisection]")
    print("   result : %.20f" % (bxmid))
    print("   iterations : %.0f" % (bi))
    print("\n[cross section]")
    print("   result : %.20f" % (kxmid))
    print("   iterations : %.0f" % (ki))

    ax2.plot([xmin], [0], 'or')
    ax2.plot([xmin+h], [0], 'or')

    # bisection result
    ax2.plot([bxmid], [0], 'ob')
    # quasi-newton(cross section) result
    ax2.plot([kxmid], [0], 'ok')

  xmin += h

np.roots(f1)
plt.show()