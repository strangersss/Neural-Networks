import numpy as np
import matplotlib.pyplot as plt
def two_moon():
    random=np.random.RandomState(0)
    i=0
    x=[]
    y=[]
    result=[]
    while i<20000:
        a=random.uniform(-1,1)
        b=random.uniform(-1,1)
        if(a*a+b*b<1)and(b*b+a*a>0.5):
            if b>0:
                x.append(a)
                y.append(b)
                result.append(1)
            if b<=0:
                x.append(a+0.5)
                y.append(b+0.2)
                result.append(-1)
            i+=1
    return x,y,result

def update():
    global X,Y,W,lr,n
    n+=1
    O=np.sign(np.dot(X,W.T))  #o是一个向量
    W_C=lr*((Y-O.T).dot(X))/int(X.shape[0])
    W=W+W_C


if __name__=='__main__':
    x,y,result=two_moon()
    X=[[1,a,b] for a,b in zip(x,y)]
    X=np.array(X)
    Y=np.array(result)
    W=(np.random.random(3)-0.5)*2

    lr=0.11
    n=0
    O=0
    for _ in range(200):
        update()
        print(W)
        print(n)
        O=np.sign(np.dot(X,W.T))
        if(O==Y.T).all():
            print("完成")
            break


    xdata=np.linspace(-2,3)
    k=- W[1]/W[2]
    b=-W[0]/W[2]
    plt.plot(x,y,'bo')
    plt.plot(xdata,xdata*k+b,'r')
    plt.show()