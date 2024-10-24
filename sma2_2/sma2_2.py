
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import math

#*********************** Pavirsius *******************************
# X,Y - meshgrid masyvai
# LF - dvieju kintamuju vektorines funkcijos vardas, 
#      argumentas paduodamas vektoriumi, isejimas vektorius ilgio 2
# rezultatas - dvigubas meshgrid masyvas Z[:][:][0:1]
def Pavirsius(X,Y,LFF):
    siz=np.shape(X)
    Z=np.zeros(shape=(siz[0],siz[1],2))
    for i in range (0,siz[0]): 
        for j in range (0,siz[1]):  Z[i,j,:]=LFF([X[i,j],Y[i,j]]).transpose();
    return Z
#*****************************************************************

#*****************************************************************
#-------- Lygciu sistemos funkcija---------
def LF(x):  # grazina reiksmiu stulpeli
    s=np.array([ (x[0]-3)**2+x[1]-8, 
                (x[0]**2+x[1]**2)/2-6*(np.cos(x[0])+np.cos(x[1]))-10])
    s.shape=(2,1)
    s=np.matrix(s)
    return s
#----------------------------------

#
#*******************  Programa ************************************ 
# 
print("Broideno metodas")

n=2 # lygciu skaicius
#x=np.matrix(np.zeros(shape=(n,1))); x[0]=2.5;x[1]=0.3

xs=3.0
ys=3.0
x=np.matrix([xs,ys]).transpose(); x[0]=xs;x[1]=ys
print(x)
# pradinis saknies artinys
maxiter=1000  # didziausias leistinas iteraciju skaicius
eps=1e-6    # reikalaujamas tikslumas
  
#------ Grafika: funkciju LF pavirsiai -----------------------------------------------------------------------
fig1=plt.figure(1,figsize=plt.figaspect(0.5));
ax1 = fig1.add_subplot(1, 2, 1, projection='3d');ax1.set_xlabel('x');ax1.set_ylabel('y');ax1.set_ylabel('z')
ax2 = fig1.add_subplot(1, 2, 2, projection='3d');ax2.set_xlabel('x');ax2.set_ylabel('y');ax2.set_ylabel('z')
plt.draw();  #plt.pause(1);
xx=np.linspace(-10,10,20);yy=np.linspace(-10,10,20);
X, Y = np.meshgrid(xx, yy); Z=Pavirsius(X,Y,LF)

wire1 = ax1.plot_wireframe(X, Y, Z[:,:,0], color='black', alpha=1, linewidth=1, antialiased=True)
surf2 = ax1.plot_surface(X, Y, Z[:,:,1], color='purple', alpha=0.4, linewidth=0.1, antialiased=True)
CS11 =  ax1.contour(X, Y, Z[:,:,0],[0],colors='b')
CS12 =  ax1.contour(X, Y, Z[:,:,1],[0],colors='g')
CS1 =   ax2.contour(X, Y, Z[:,:,0],[0],colors='b')
CS2 =   ax2.contour(X, Y, Z[:,:,1],[0],colors='g')

XX=np.linspace(-10,10,2);  YY=XX; XX, YY = np.meshgrid(XX, YY); ZZ=XX*0
zeroplane = ax2.plot_surface(XX, YY, ZZ, color='gray', alpha=0.4, linewidth=0, antialiased=True)
#---------------------------------------------------------------------------------------------------------------

dx=0.1   # dx pradiniam Jakobio matricos iverciui
A=np.matrix(np.zeros(shape=(n,n)))
x1=np.zeros(shape=(n,1));
for i in range (0,n):   
   x1=np.matrix(x);
   x1[i]+=dx;
   A[:,i]=(LF(x1)-LF(x))/dx

ff=LF(x)
ax1.plot3D([x[0,0],x[0,0]],[x[1,0],x[1,0]],[0,ff[0,0]],"m*-")
plt.draw();


print(A)
print(ff.transpose(),"Pradine funkcijos reiksme")

# Function to compare solutions
def compare_solutions(A, b, custom_solution):
    try:
        np_solution = np.linalg.solve(A, b)
        print("\nComparing custom solution with numpy solution:")
        print("Custom solution x:")
        print(custom_solution)
        print("Numpy solution x:")
        print(np_solution)
        difference = np.linalg.norm(custom_solution - np_solution)
        print(f"Difference between solutions: {difference}")
    except np.linalg.LinAlgError as e:
        print(f"Error in numpy solution: {e}")

for i in range (1,maxiter):
    deltax=-np.linalg.solve(A,ff); 
    x1=np.matrix(x+deltax); 
    ff1=LF(x1)
    A+=(ff1-ff-A*deltax)*deltax.transpose()/(deltax.transpose()*deltax);  
    tiksl=np.linalg.norm(x-x1)+np.linalg.norm(ff1); print(tiksl,"tikslumas")
    ff= ff1; 
    x=x1; 
    if tiksl < eps: break; 
    else: 
        print(x1.transpose(),"x1 ")
   #------ Grafika:  -----------------------------------------------------------------------------------------------
    ax1.plot3D([x[0,0],x1[0,0]],[x[1,0],x1[1,0]],[0,0],"ro-")  # reikia prideti antra indeksa, kadangi x yra matrica
    ax1.plot3D([x[0,0],x1[0,0]],[x[1,0],x1[1,0]],[ff[0,0],ff1[0,0]],"c-.")
    ax1.plot3D([x1[0,0],x1[0,0]],[x1[1,0],x1[1,0]],[0,ff1[0,0]],"m*-")
    plt.draw();
    #---------------------------------------------------------------------------------------------------------------
ax1.plot3D([x[0,0],x[0,0]],[x[1,0],x[1,0]],[0,0],"ks")
compare_solutions(A, ff, x1)

print(x1.transpose(),"Sprendinys")
print(ff1,"funkcijos reiksme")
print(tiksl,"Galutinis tikslumas")
# Define the colors and initialize plot for marking solutions
colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow', 'black']
solutions = []
color_index = 0

# Create a grid ranging from -10 to 10 with a step of 1
xx = np.arange(-10, 11, 1)
yy = np.arange(-10, 11, 1)
X, Y = np.meshgrid(xx, yy)

# Iterate over the grid points
for i in range(len(xx)):
    for j in range(len(yy)):
        xs = xx[i]
        ys = yy[j]
        print (str(xs) + " ; " + str(ys))
        x = np.matrix([xs, ys], dtype=float).transpose()
        
        # Reset Jacobian approximation
        for k in range(n):
            x1 = np.matrix(x, dtype=float)
            x1[k] += dx
            A[:, k] = (LF(x1) - LF(x)) / dx
        ff = LF(x)
        
        for k in range(maxiter):
            deltax = -np.linalg.solve(A, ff)
            x1 = np.matrix(x + deltax, dtype=float)
            ff1 = LF(x1)
            A += (ff1 - ff - A * deltax) * deltax.transpose() / (deltax.transpose() * deltax)
            tiksl = np.linalg.norm(deltax) + np.linalg.norm(ff1)
            ff = ff1
            x = x1
            if tiksl < eps:
                break
            #ax2.plot3D([x[0,0],x1[0,0]],[x[1,0],x1[1,0]],[0,0],"ro-")
            plt.draw();
        
        # Check if the solution is close to an existing one
        solution_found = False
        for idx, sol in enumerate(solutions):
            if np.allclose(sol, x, atol=1e-6):
                solution_found = True
                color_idx = idx
                break
        
        if not solution_found:
            print("================================naujas===================================")
            color_index = len(solutions) % len(colors)
            solutions.append(x)
            print(color_index)
            color_idx = color_index
        
        ax2.plot(xs, ys, 'o', color=colors[color_idx])
        print(x1.transpose(),"Sprendinys")
        print(ff1,"funkcijos reiksme")
        print(tiksl,"Galutinis tikslumas")


for i in range(0, len(solutions)):
    print("jei su is grido paimta liestine gautas sprendinys: "+ str(solutions[i][0])+ "; " + str(solutions[i][1]) +" \njo spalva yra: " + colors[ i % len(colors)])

#------ Grafika:  ---------------------------------------
plt.draw();
#--------------------------------------------------------


plt.show();


#str1=input("Baigti darba? Press 'Y' \n")
