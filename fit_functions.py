def model(x, D, R, C, n,Y0,X0):
    return D / (R * (1 / (C * (x)**n + 1)) + 1)+Y0

def hillEq(x,Kd,n):
    return (x**n)/(Kd**n+x**n)