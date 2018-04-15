from moon import moon
import matplotlib.pyplot as plt
import numpy as np

def update(X_i,Y_i,lr):
    global W
    O=np.dot(X_i,W.T)
    W_C=lr*(Y_i-O)*X_i
    W=W+W_C
def error(X,W,Y):
    pre=np.sign(np.dot(X,W.T))
    return np.linalg.norm(pre - Y)
def self_adp(n):
    return 0.001/(1+n/5)
def train(X,Y,e,num):
    global W
    #lr = 0.0001  # 学习率
    O = 0  # 预测结果
    for n in range(num):
        lr = self_adp(n)
        for i in range(N):
            update(X[i], Y[i],lr)
        e.append(error(X, W, Y))
        O = np.sign(np.dot(X, W.T))  # 训练模型预测
        if (O == Y.T).all():  # 模型训练完成
            print (n)
            break
    return n
if __name__=='__main__':
    num = 1000  # 迭代次数
    N = 500  # 点个数
    Moon=moon(0.5,0.1,N)
    x,y,result=Moon.create()
    X=[[1,a,b] for a,b in zip(x,y)]
    X=np.array(X)
    Y=np.array(result)
    #W=(np.random.random(3)-0.5)*2  #初始为随机小数
    W=np.array([0,0,0])#初始值为0
    e = []  # 误差
    n=train(X,Y,e,num)#训练
    print(W)
    xdata=np.linspace(-2,3)
    k=- W[1]/W[2]
    b=-W[0]/W[2]
    plt.subplot(121)
    plt.plot(x,y,'bo')
    plt.plot(xdata,xdata*k+b,'r')
    plt.subplot(122)
    plt.plot(np.linspace(1,n+1,n+1),e)
    plt.show()