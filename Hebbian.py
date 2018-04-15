import numpy as np
import matplotlib.pyplot as plt
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
            y.append(b+0.5)
            result.append(1)
        if b<=0:
            x.append(a)
            y.append(b-0.5)
            result.append(-1)
        i+=1
X=np.array(list(zip(x,y)))
Y=np.array(result)
W=(np.random.random(2)-0.5)*2

lr=0.5
out=0
b=0
for i in range(20000):
    out=np.sign(np.dot(X[i],W.T))
    W_C=lr*out*X[i]+b
    W+=W_C
print(W)

k=-W[0]/W[1]
xdata=np.linspace(-2,2)
plt.plot(x,y,'bo')
plt.plot(xdata,xdata*k,'r')
plt.show()