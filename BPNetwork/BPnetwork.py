import random
import numpy as np
class BPnetwork():
    def __init__(self,sizes,fun,error,eta,eta_is_change=False,w_is_auto=True,w=[],num=100,boundary=0):
        """
                :param sizes: list类型，储存每层神经网络的神经元数目
                              譬如说：sizes = [2, 3, 2] 表示输入层有两个神经元、
                              隐藏层有3个神经元以及输出层有2个神经元
                :fun:字符类型，选择激活函数，logistic或者atan
                :w_is_auto:bool类型，是否手动配置初始权值，true表示随机生成，默认true
                :w:list类型，w_is_auto为false时，不可为空
                :num: 迭代回合数，默认100次
                :eta:学习率
                :error: 误差要求
                :boundary:判别阈值，默认为0
                :eta_is_auto:学习率是否可变，若为true则eta为初始学习率，false则eta为固定学习率
                """
        # 神经网络层数
        self.num_layers=len(sizes)
        self.sizes=sizes
        # 生成初始权值
        if w_is_auto==True:
            self.weights = [(np.random.randn(y, x)-0.5)*2
                            for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.weights=[np.array(i) for i in w]
        self.fun=fun
        self.num=num
        self.boundary=boundary
        self.eta=eta
        self.eta_is_change=eta_is_change
        self.error=error
    def function(self,num_in):
        """
                选择激活函数
                :param num_in: 输入值
                :return: 通过激活函数后的值
                """
        if self.fun=='logistic':
            a=2#logistic函数中参数的配置，默认为1
            return 1/(1+np.exp(-num_in*a))-(0.5-self.boundary)
        if self.fun=='atan':
            a=1
            b=2#atan函数中参数的配置，默认为1
            return a*(1-np.exp(-b*num_in))/(1+np.exp(-b*num_in))
    def fun_prime(self,num_in):
        """
                        激活函数的导数
                        :param num_in: 输入值
                        :return: 求导后的值
                        """
        if self.fun=='logistic':
            a=2#与原函数参数配置相同
            return a*num_in*(1-num_in)
        if self.fun=='atan':
            a=1
            b=2
            return b/a*(a-num_in)*(a+num_in)
    def cost(self,output,y):
        #计算误差
        return output-y
    def forward(self,x):
        """
                       计算向前传播每个神经元的值
                        :param x: 输入层输入值
                        :return: 计算后输出层的值
                        """
        for  w in self.weights:
            # 加权求和以及加上 biase
            x= self.function(np.dot(x,w.T))
        return x

    def backward(self,x,y,n=1):
        """"
        向后传播更新权值
        :param
        x: 输入层输入值
        y:预期结果
        n:第n次训练
        """
        if self.eta_is_change:
            eta=self.eta/(1+n/5)
        else:
            eta=self.eta
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        #前向
        activation=np.array([x])
        activations=[activation]#储存每层神经元的值
        zs=[]#储存未经激活函数计算的神经元的值
        for w in self.weights:
            z=np.dot(activation,w.T)
            zs.append(z)
            activation=self.function(z)
            activations.append(activation)
        #输出层到隐藏层更新
        #求δ的值
        delta=self.cost(activations[-1],y)*self.fun_prime(activations[-1])
        nabla_w[-1]=np.dot(delta.T,activations[-2])
        for l in range(2,self.num_layers):
            z=activations[-l]
            sp=self.fun_prime(z)
            delta=np.dot(delta,self.weights[-l+1])*sp
            nabla_w[-l]=np.dot(delta.T,activations[-l-1])
        self.weights=[w-eta*nw for w,nw in zip(self.weights,nabla_w)]
    def get_w(self):
        return self.weights
    def test(self,test_data):
        pre=[]
        for i in self.forward(test_data):
            if i[0] >= self.boundary:
                pre.append(1)
            else:
                pre.append(-1)
        return pre
    def run(self,train_data,result):
        err=[]
        n=self.num

        for j in range(self.num):
            for i  in range(len(train_data)):
                self.backward(train_data[i],result[i],j)
            pre=self.test(train_data)
            err.append(np.linalg.norm(np.array(pre) - np.array(result))/len(train_data))
            if (np.linalg.norm(np.array(pre) - np.array(result))/len(train_data)) <= self.error:
                self.get_w()
                n=j+1
                break
        return self.get_w(),pre,n,err














