from moon import moon
from BPnetwork import BPnetwork
import matplotlib.pyplot as plt
import numpy as np
if __name__=='__main__':
    N=500
    Moon=moon(0.6,0.5,N)
    x,y,result=Moon.create()
    X=[[a,b] for a,b in zip(x,y)]
    sizes=[2,20,1]
    fun='atan'
    error=0
    eta=0.001
    w_is_auto = True
    num=1500
    boundary = 0
    BP=BPnetwork(sizes,fun,error,eta,eta_is_change=False,num=num,boundary=boundary)
    w,pre,n,err=BP.run(X,result)
    print(w)
    print(n)
    plt.subplot(221)
    for i in range(N-1):
        if pre[i]>boundary:
            plt.plot(x[i], y[i], 'ro')
        else:
            plt.plot(x[i], y[i], 'bo')
    plt.subplot(222)
    plt.plot(np.linspace(1, n, n), err)
    plt.subplot(223)
    x=(np.random.rand(1000)-0.5)*4
    y=(np.random.rand(1000)-0.5)*4
    Input=[[a,b] for a,b in zip(x,y)]
    Output=BP.test(Input)
    for i in range(1000):
        if Output[i]>boundary:
            plt.plot(x[i], y[i], 'ro')
        else:
            plt.plot(x[i], y[i], 'bo')
    plt.show()
