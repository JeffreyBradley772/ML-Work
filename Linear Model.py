"""

Jeffrey Bradley
9/5/2020

"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.linspace(0, 2, 100)
y = 3 * x + 0.9 + np.random.randn(*x.shape) * 0.3

##plt.scatter(x,y)
##plt.show()

class LinearModel(object):
    
    def __init__(self):
        self.w0 = tf.Variable(5.0)
        self.bias = tf.Variable(4.0)
        
        #self.yhatList = []

    def __call__(self, x): #-> trying to generate y^
        
        return (self.w0 * x) + self.bias

        

def loss(yActual, yModel):
    loss = tf.reduce_mean(tf.square(y - yModel))
    return loss

def train(Model, x,y,lr = 0.05):
    with tf.GradientTape() as t:
        modelLoss = loss(y,Model(x))
##    print(modelLoss)
##    print(Model.w0,Model.bias)
    w0Change,biasChange = t.gradient(modelLoss,[Model.w0,Model.bias])
##    print(w0Change,biasChange)
    
    Model.w0.assign_sub(w0Change * lr)
    Model.bias.assign_sub(biasChange * lr)
##        Find the amount you should adjust the two parameters based on the loss c.
##        Multiply the amount you should change the parameters
##        by the learning rate (learning rate is hard coded at 0.05)
##        and replace the parameter
            

def main():
    w0_s =[]
    bias_s =[]
    linear = LinearModel()
    #yhat = linear.__call__(x)
    for i in range(1,101):
        w0_s.append(linear.w0)
        bias_s.append(linear.bias)
        yhat = linear(x)
        Loss = loss(y,yhat)
        train(linear,x,y)
        print(Loss.numpy())
        
    print('Weight: ' + str(linear.w0.numpy()))
    print('Bias: ' + str(linear.bias.numpy()))
    print('Final Loss: ' + str(loss(y,yhat).numpy()))
    
    yhat = linear(x)
    plt.scatter(x,y)
    plt.plot(x,yhat)
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.legend(['Model Prediction','Actual Data'])
    plt.title('Assignment 2 Data')
    plt.show()

    

    
    
        
        
    
if __name__ == '__main__':
    main()
        
