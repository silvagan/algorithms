import numpy as np
import sys

def SpM(*args):
#******************** Isveda matrica ******************** 
# args[0] - isvedamas kintamasis 
# args[1] - kintamojo vardas (neprivalomas)
    A=args[0]; str="";
    if len(args) == 2: str=args[1];
    else:
        for name, value in globals().items():
            if value is args[0]: str=name; break
    
    if np.isscalar(A): print(str,"=",A); return
    else: siz=np.shape(A);
    if len(siz) > 1 & siz[0] == 1: print(str,"=",A);
    else:  print(str,"="); print(A);
    return
#******************************************************************
#******************************************************************    
#lygciu sistema 1      
A1 = np.matrix([
    [3, 7, 1, 3],
    [1, -6, 6, 9],
    [4, 4, -7, 1],
    [-3, 8, 2, 1]],dtype=float)        # koeficientu matrica
b1=np.matrix([37,11,38,0],dtype=float).transpose()   #laisvuju nariu vektorius-stulpelis

#lygciu sistema 14
A2 = np.matrix([
    [2, 4, 6, -2],
    [1, 3, 1, -3],
    [1, 1, 5, 1],
    [2, 3, -3, -2]],dtype=float)        # koeficientu matrica
b2=np.matrix([4,-7,11,-4],dtype=float).transpose()   #laisvuju nariu vektorius-stulpelis

#lygciu sistema 20
A3 = np.matrix([
    [2, 4, 6, -2],
    [1, 3, 1, -3],
    [1, 1, 5, 1],
    [2, 3, -3, -2]],dtype=float)        # koeficientu matrica
b3=np.matrix([2,1,7,2],dtype=float).transpose()   #laisvuju nariu vektorius-stulpelis


def Solve(A, b, name):

    print("-----------------------")
    print("sprendziama matrica: " + name)
    print("-----------------------")
    Ap=np.matrix(A)   # bus naudojama patikrinimui
    n=(np.shape(A))[0]   # lygciu skaicius nustatomas pagal ivesta matrica A
    nb=(np.shape(b))[1]  # laisvuju nariu vektoriu skaicius nustatomas pagal ivesta matrica b


    SpM(A);SpM(b);SpM(n);SpM(nb);


    # tiesioginis etapas(QR skaida):
    Q=np.identity(n)
    for i in range (0,n-1):
        print("\n---------------")
        print("ciklo iteracija: " + str(i))
        print("---------------")
        #sukame cikla kiekvienai matricos eilutei
        #z nustatytas taip, kad jame butu dabartines eilutes elementai
        z=A[i:n,i];                                     SpM(z, "z"); print("")
        #vektorius nuliu
        zp=np.zeros(np.shape(z)); 
        #nustatome pirmaji zp elementa kaip z ilgi
        zp[0]=np.linalg.norm(z);                        SpM(zp, "zp"); print("")

        #skaiciuojame omega kuri yra skirtumas tarp z ir zp
        omega=z-zp;
        #pabaigiame realizuoti formule
        omega=omega/np.linalg.norm(omega);              SpM(omega, "omega"); print("")

        #sudarome householder matrica Qi (identity matrix dydzio (n-i) pakeista omegos)
        Qi=np.identity(n-i)-2*omega*omega.transpose();  SpM(Qi, "Qi"); print("")

        #atnaujinti A nuo dabartines eilutes tolyn dauginant su Qi
        A[i:n,i:n]=Qi.dot(A[i:n,i:n]);                  SpM(A, "A"); print("")
        #naujiname Q taikydami Qi nuo dabartinio stulpelio tolyn
        Q[:,i:n]=Q[:,i:n].dot(Qi);                      SpM(Q, "Q"); print("")
    # po ciklo mes turime virsutine trikampe matrica A(R) ir matrica Q kuri sukaupe ortogonalias transformacijas

    print("\nmatrica Q ir trikampe matrica A(R) sekmingai sumontuota")

    if np.any(np.abs(np.diag(A)) < 1e-10):  # Tikrinimas ar istrizai einantys matricos R elementai yra netoli nulio
        rank_A = np.linalg.matrix_rank(A)
        rank_Ab = np.linalg.matrix_rank(np.hstack((Ap, b)))
        if rank_A < rank_Ab:
            print("sistema neturi sprendiniu.\n")
        elif rank_A < n:
            print("sistema turi daug sprendiniu.\n")
        return
    SpM(A)

    print("tiesioginio etapo pabaiga\n")

    # atgalinis etapas:
    b1=Q.transpose().dot(b);
    x=np.zeros(shape=(n,nb));
    for i in range (n-1,-1,-1):    # range pradeda n-1 ir baigia 0 (trecias parametras yra zingsnis)
        x[i,:]=(b1[i,:]-A[i,i+1:n]*x[i+1:n,:])/A[i,i];
        SpM(x, "x")

    SpM(x,'sprendinys:');
    print("------------ sprendinio patikrinimas ----------------");
    liekana=Ap.dot(x)-b; SpM(liekana, "liekana");
    SpM(np.linalg.norm(liekana)/ np.linalg.norm(x),"bendra santykine paklaida:")

def check_solution_qr(A, b):
    # QR Decomposition
    Q, R = np.linalg.qr(A)
    
    # Solve R * y = Q.T * b
    y = np.dot(Q.T, b)
    x_builtin = np.linalg.solve(R, y)
    print(f"check results: {x_builtin}\n")

# Solve the systems
x1 = Solve(A1, b1, "A1")
check_solution_qr(A1, b1)
x2 = Solve(A2, b2, "A2")
check_solution_qr(A2, b2)
x3 = Solve(A3, b3, "A3")
check_solution_qr(A3, b3)

