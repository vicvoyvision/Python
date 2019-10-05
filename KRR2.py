# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 20:10:27 2019

@author: vicvo
"""

# -*- coding: utf-8 -*-



import os
#os.chdir('D:/STUDY/RHUL/project')
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei']
import tkinter as tk
from tkinter import ttk


import sys
import tkinter as Tk
import matplotlib
from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

class ridge():
    '''
    x_train:train set x
    y_train:train set y
    lamda：ridge parameter
    rbf_var:The kernel function is gaussian kernel function and the parameter is kernel width'''
    def __init__(self,x_train,y_train,lamda=0,rbf_var=1,a=1,t=1,c=2,d=1):
        X = x_train
        Y = y_train
        [n,m] = X.shape
        self.x_mean = np.mean(X,0)
        self.x_std = np.std(X,0)
        self.y_mean = np.mean(Y,0)
        X = (X-self.x_mean)/self.x_std
        self.X = np.column_stack((X,np.ones(n)))
        self.Y = Y
        self.lamda = lamda
        self.rbf_var=rbf_var
        
        self.a=a
        self.t=t
        self.c=c
        self.d=d
        
        
        
    def RR(self,x_test):
        '''Ridge regression method, input test data set, output forecast data
        x_test:test set
        y_predict: predict value'''
        I = np.identity(self.X.shape[1])
        w = np.linalg.inv(self.X.T@self.X+self.lamda*I)@self.X.T@self.Y#回归系数计算
        [p,q] = x_test.shape
        x = (x_test-self.x_mean)/self.x_std
        x = np.column_stack((x,np.ones(p)))
        y_predict = x@w
        return y_predict
    def KRR(self,x_test,m=0):
  
        [p,q] = x_test.shape
        x = (x_test-self.x_mean)/self.x_std  
        x = np.column_stack((x,np.ones(p)))    
        if m==1:
            K =self.rbfkernel_function(self.X,self.X)
        elif m==0:
            K =self.polynomial_kernel(self.X,self.X,self.a,self.t,self.c,self.d)
        I = np.identity(K.shape[0])
        a = np.linalg.inv(self.lamda*I+K)@self.Y
        if m==1:
            y_predict = self.rbfkernel_function(x,self.X)@a
        elif m==0:    
            y_predict = self.polynomial_kernel(x,self.X)@a
        return y_predict
        
        
    def rbfkernel_function(self,x,y):
        '''Instead of the inner product, the kernel is computed, which is the gaussian kernel'''
        K = np.sum(x*x,axis=1,keepdims=True)+np.sum(y*y,axis=1,keepdims=True).T
        K = K-2*x@y.T
        K = K/(2*self.rbf_var**2)    
        K = np.exp(-K)
        return K
    
    def polynomial_kernel(self,x,y,a=1,t=1,c=2,d=1):
        K=(self.a*np.dot(x**self.t,y.T)+self.c)**self.d
        return K
        
    
    
        
    
if __name__== '__main__':
    matplotlib.use('TkAgg')
   
   
    window = tk.Tk()
    window.title('Choose parameter')
    window.geometry('260x260')
    #window.configure(background='grey')
    var = tk.StringVar()
    la = tk.Label(window,
                  text='Choose KNN lambda',
                  font=('Arial', 12), width=30, height=2)
    la.grid(column=1, row=1)
    
    def click():
        window.destroy()
    
    number = tk.StringVar()
    numberChosen = ttk.Combobox(window, width=12, textvariable=number)
    numberChosen['values'] = (0.01,0.1,1,10)     
    numberChosen.grid(column=1, row=3)      
    numberChosen.current(0)   
    
    l2=tk.Label(window, text='Choose a kernel function', font=('Arial', 12), width=30, height=2)
    l2.grid(column=1, row=5)

    kernel = tk.StringVar()
    kernelChosen = ttk.Combobox(window, width=12, textvariable=kernel)
    kernelChosen['values'] = ('rbf','ploy')     
    kernelChosen.grid(column=1, row=7)      
    kernelChosen.current(0) 
    
    l3=tk.Label(window, text='Choose The data set', font=('Arial', 12), width=30, height=2)
    l3.grid(column=1, row=9)
    
    ds = tk.StringVar()
    dsChosen = ttk.Combobox(window, width=12, textvariable=ds)
    dsChosen['values'] = ('Boston Housing','Air Quality')     
    dsChosen.grid(column=1, row=11)      
    dsChosen.current(0) 
    
    b1 = tk.Button(window, text='close', command=click)
    b1.place(x=110, y=210, anchor=tk.NW)
    
    window.mainloop()
    para=float(number.get())
    kernel=kernel.get()
    ds=ds.get()
    
    if ds=='Boston Housing':
        
    
        loaded_data = datasets.load_boston()
        data_X = loaded_data.data
        data_y = loaded_data.target
        T=data_X.shape[0]
        n=data_X.shape[1]
    
    else:
        
        air=pd.read_excel('AirQualityUCI.xlsx')
        air.replace(to_replace= -200, value= np.NaN, inplace= True)
        air.drop(['NMHC(GT)'], axis= 1, inplace= True)
        air["T"] = air.groupby("Date")["T"].transform(lambda x: x.fillna(x.mean()))
        air["CO(GT)"] = air.groupby("Date")["CO(GT)"].transform(lambda x: x.fillna(x.mean()))
        air["NOx(GT)"] = air.groupby("Date")["NOx(GT)"].transform(lambda x: x.fillna(x.mean()))
        air["NO2(GT)"] = air.groupby("Date")["NO2(GT)"].transform(lambda x: x.fillna(x.mean()))
        air["PT08.S4(NO2)"] = air.groupby("Date")["PT08.S4(NO2)"].transform(lambda x: x.fillna(x.mean()))
        air["PT08.S5(O3)"] = air.groupby("Date")["PT08.S5(O3)"].transform(lambda x: x.fillna(x.mean()))
        air["RH"] = air.groupby("Date")["RH"].transform(lambda x: x.fillna(x.mean()))
        air["PT08.S3(NOx)"] = air.groupby("Date")["PT08.S3(NOx)"].transform(lambda x: x.fillna(x.mean()))
        air["PT08.S2(NMHC)"] = air.groupby("Date")["PT08.S2(NMHC)"].transform(lambda x: x.fillna(x.mean()))
        air["C6H6(GT)"] = air.groupby("Date")["C6H6(GT)"].transform(lambda x: x.fillna(x.mean()))
        air.fillna(method='ffill', inplace= True)
        data_y=air['AH'].values
        a=air
        a.drop(a.columns[[0,1,13]], axis=1, inplace=True)
        data_X=a.values
        T=data_X.shape[0]#row
        n=data_X.shape[1]#col
      
     

   
    #Randomly divide training set, verification set and test set
    index = np.random.permutation(range(T))
    data_train_X = data_X[index[:int(T/2)],:]
    data_train_y = data_y[index[:int(T/2)]]
    data_validation_X = data_X[index[int(T/2):int(T*0.75)],:]
    data_validation_y = data_y[index[int(T/2):int(T*0.75)]]
    data_test_X = data_X[index[int(T*0.75):],:]
    data_test_y = data_y[index[int(T*0.75):]]
    '''
   
    index = np.random.permutation(range(T))
    data_train_X = data_X[index[:int(T*0.75)],:]
    data_train_y = data_y[index[:int(T*0.75)]]
    data_validation_X = data_X[index[int(T*0.75):int(T*0.95)],:]
    data_validation_y = data_y[index[int(T*0.75):int(T*0.95)]]
    data_test_X = data_X[index[int(T*0.75):],:]
    data_test_y = data_y[index[int(T*0.75):]]
     '''
    '''
    
    index = np.random.permutation(range(506))
    data_train_X = data_X[index[:250],:]
    data_train_y = data_y[index[:250]]
    data_validation_X = data_X[index[250:350],:]
    data_validation_y = data_y[index[250:350]]
    data_test_X = data_X[index[350:],:]
    data_test_y = data_y[index[350:]]
     '''
   #Training model and prediction, super parameter is obtained by manual debugging

    para=float(para)
    model = ridge(data_train_X,data_train_y,lamda=para,rbf_var=1,a=1,t=2,c=2,d=1)
    if kernel=='rbf':
        y_predict = model.KRR(data_test_X,1)
    else:
        y_predict = model.KRR(data_test_X,0)
    

    y_ = model.RR(data_test_X) #RR
    
    MSERR=np.mean(pow(data_test_y - y_,2))
    MSEKRR=np.mean(pow(data_test_y - y_predict,2))
    
    #visulazation
    plt.figure()
    plt.rcParams['figure.figsize'] = (10.0, 8.0)
    plt.subplot(2,1,1)
    plt.plot(data_test_y,'r',label=u'True Y') 
    plt.plot(y_predict,'b',label=u'Predicted Y')
    plt.title(u'Kernel Ridge Regression')
    plt.ylabel(u'Y')
    plt.legend(loc=1)
    plt.subplot(2,1,2)
    plt.plot(data_test_y,'r',label=u'True Y')
    plt.plot(y_,'g',label=u'Predicted Y')
    plt.title(u'Ridge Regression')
    plt.xlabel(u'Sample No.')
    plt.ylabel(u'Y')
    plt.legend(loc=1)
    plt.show()
    
    

    plt.scatter(data_test_y, y_)  # Scatter plot, ridge regression
    plt.ylabel(u'Predicted Y')
    plt.xlabel(u'True Y')
    plt.title(u'Ridge Regression')
   


    plt.plot(data_test_y, data_test_y, 'k-')
    plt.show()
    
    plt.scatter(data_test_y, y_predict)   
    plt.ylabel(u'Predicted Y')
    plt.xlabel(u'True Y')
    plt.title(u'Kernel Ridge Regression')
    plt.plot(data_test_y, data_test_y, 'b-') #line y=x

    
    plt.plot(data_test_y, data_test_y, 'k-')
    plt.show()
    
    print('The data you choose is:',ds)
    print('MSE of Ridge Regression is:',MSERR)
    print('MSE of Kernel Ridge Regression is:',MSEKRR)
    print('lambda is :',para)
    print('kernel is :',kernel)
    
    
   
    print("Kernel method parameters and MSE:")
    # choose 1 as gaussian kernel and 0 as polynomial kernel
    rbf=[]
    poly=[]
    L=[]
    p=[]
    # choose 1 as gaussian kernel and 0 as polynomial kernel
    for i in np.arange(0.01,1.01,0.05):
        para=round(i, 2)#lambda
        model = ridge(data_train_X,data_train_y,lamda=para,rbf_var=2,a=1,t=2,c=2,d=1)
        y_predict0 = model.KRR(data_test_X,0) #poly
        
        y_predict1 = model.KRR(data_test_X,1)
        
        
        MSEKRR1=np.mean(pow(data_test_y - y_predict1,2))
        rbf.append(MSEKRR1)
        
        MSEKRR2=np.mean(pow(data_test_y - y_predict0,2))
        
        
        poly.append(MSEKRR2)
        
        
        model1 = ridge(data_train_X,data_train_y,lamda=para,rbf_var=2,a=1,t=1,c=2,d=1)
        y_predict3 = model1.KRR(data_test_X,0) #poly
        MSEKRR3=np.mean(pow(data_test_y - y_predict3,2))
        p.append(MSEKRR3)
        
        
        
        L.append(para)
    plt.figure()
    plt.ylabel(u'MSE')
    plt.xlabel(u'Lambda')
    plt.title(u'Kernel Ridge Regression')
    plt.plot(L, rbf, 'g-', label='RBF  ')
    plt.plot(L, poly, 'r--', label='Poly  t=2 ')
    plt.plot(L, p, 'b--', label='Poly t=1')
    plt.legend() # 显示图例
    plt.show()

     
    
    
    
    
   
    

    
    
    
    
    #2
    
    
    
   
   
    
    root =Tk.Tk()
  
    root.title("matplotlib in TK")
    #Set graphics size and quality
    f =Figure(figsize=(5,4), dpi=100)
    a = f.add_subplot(111)
    #t = arange(0.0,3,0.01)
    #s = sin(2*pi*t)
    #draw
    #a.plot(t, s)
    a.scatter(data_test_y, y_predict)   
   # a.ylabel(u'Predicted Y')
    #a.xlabel(u'True Y')
    #a.title(u'Kernel Ridge Regression')
    a.plot(data_test_y, data_test_y, 'b-') #line y=x

    a.plot(data_test_y, data_test_y, 'k-')
    #plt.show()

    
    #Displays the drawn graph to the tkinter window
    
    canvas =FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    #Display the matplotlib graphing navigation toolbar to the tkinter window
    toolbar =NavigationToolbar2TkAgg(canvas, root)
    toolbar.update()
    canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    #Defines and binds keyboard event handlers
    def on_key_event(event):
        
        print('you pressed %s'% event.key)
        key_press_handler(event, canvas, toolbar)
    canvas.mpl_connect('key_press_event', on_key_event)
    #Button to click the event handler
    def _quit():
    #End the event main loop and destroy the application window
        root.quit()
        root.destroy()
    button =Tk.Button(master=root, text='Quit', command=_quit)
    button.pack(side=Tk.BOTTOM)
    Tk.mainloop()
    
    
