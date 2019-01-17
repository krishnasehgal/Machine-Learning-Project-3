
# coding: utf-8

# ## Create MNIST datasets- TRAINING, VALIDATION AND TESTING 

# In[ ]:


def MNIST():
    
    with open('/Users/krishna/Downloads/mnist.pkl', 'rb') as f:
        training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
        f.close()
    
    training_x=np.array(training_data[0])
    training_y=np.array(training_data[1])
    validation_x=np.array(validation_data[0])
    validation_y=np.array(validation_data[1])
    test_x=np.array(test_data[0])
    test_y=np.array(test_data[1])
    return training_x,training_y,test_x,test_y 


# ## CREATING USPS TRAINING DATASET

# In[ ]:


def USPS():
    
    USPSMat  = []
    USPSTar  = []
    curPath  = '/Users/krishna/Downloads/USPSdata/Numerals'
    savedImg = []
    new=[]
    training_percent=0.8
    testing_percent=0.2
    for j in range(0,10):
        curFolderPath = curPath + '/' + str(j)
        imgs =  os.listdir(curFolderPath)
        for img in imgs:
            curImg = curFolderPath + '/' + img
            if curImg[-3:] == 'png':
                img = Image.open(curImg,'r')
                img = img.resize((28, 28))
                savedImg = img
                imgdata = (255-np.array(img.getdata()))/255
                USPSMat.append(imgdata)
                USPSTar.append(j)
    testing_x=np.array(USPSMat)
    testing_y=np.array(USPSTar)

    return testing_x, testing_y


# ## Logistic Regression

# In[ ]:


def derivative(training_data, calculated_y, expected_y):
    m=len(training_data)
    delta=(1/m)*np.dot((calculated_y-expected_y),training_data)
    return delta


# In[ ]:


def softmax(mat):
    e_x = np.exp(mat - np.max(mat))
    a= (e_x / e_x.sum(axis=0))
    return a 


# In[ ]:


def accuracy_logistic(calc_output, target): 
    counter=0
    for i,j in zip(calc_output,target):
        if(i==j):
            counter+=1
    return (counter/len(calc_output))
    


# In[ ]:


def logistic_regression(Epochs, training_x, training_y, testing_x, testing_y ,Learning_rate):
    
    CLASSES=10
    hot_training_y=(encodeLabel(training_y)).T
    
    w=np.random.rand(CLASSES,training_x.shape[1])
    
    for i in range(0,Epochs):
        wtx = np.dot(w,np.transpose(training_x))
        a=softmax(wtx)
        
        deltaw=derivative(training_x, calculated_y=a, expected_y=hot_training_y)
        
        w=w-Learning_rate*deltaw
        
        
    
    print("wTx.shape: {0}, A.shape: {1}, deltaw.shape: {2}, w.shape: {3}".format(wtx.shape, a.shape, deltaw.shape, w.shape))
    
    
    print("***************LOGISTIC REGRESSION***************")
    print("NUMBER OF EPOCHS",Epochs)
    print("Learning Rate ", Learning_rate)
    
    t=np.dot(w,np.transpose(testing_x))
    
    
    
    result=softmax(t)
    
    c=np.argmax(result,axis=0)
    
    
    acc_mnist=accuracy_logistic(c,testing_y)
    
    confusion_mat_mnist= confusion_matrix(testing_y, c)
    
    print("-----------MNIST--------------")
    print("accuracy of MNIST is ")
    print(acc_mnist)
    print("confusion matrix")
    print(confusion_mat_mnist)
    return w
    


# # TESTING LOGISTIC REG. ON USPS DATASET

# In[ ]:


def logistic_usps(testing_x, testing_y, w):
    t=np.dot(w,np.transpose(testing_x))
    result=softmax(t)
    
    print("-----------USPS Data-Set--------------")
    c=np.argmax(result,axis=0)
    acc_usps=accuracy_logistic(c, testing_y)
    
    print("accuracy for USPS is ")
    print(acc_usps)
    
    confusion_mat_usps=confusion_matrix(testing_y, c)
    print("confusion matrix")
    print(confusion_mat_usps)
    


# ## MAIN FUNCTION

# In[ ]:


def main():
    
    epoch=700
    LR=0.25
    
    x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist= MNIST()
    x_test_usps, y_test_usps = USPS()  
     
     #logistic regression
    final_weights=logistic_regression(epoch, x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist,LR)
    logistic_usps(x_test_usps,y_test_usps, final_weights)
    
    # training neural network with mnist dataset and returning with the model
    model_neural=neural_network(x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist)
    
    # Testing the neural network model with usps dataset
    neural_test(x_test_usps,y_test_usps, model_neural)
    
    random_forest(x_train_mnist, y_train_mnist, x_test_mnist, y_test_mnist,x_test_usps,y_test_usps)


    model=SVM(x_train_mnist, y_train_mnist)
    
    svm_test(x_test_mnist,y_test_mnist,x_test_usps, y_test_usps, model)
    
    


# In[ ]:


def SVM(train_x,train_y):
    model= SVC(kernel='rbf', C=2, gamma = 0.05)
    model.fit(train_x,train_y)
    return model
    


# In[ ]:


def svm_test(x_test,y_test,x_testusps,y_testusps, model):
    calculated = model.predict(x_test)
    counter=0
    for i, j in zip(calculated,y_test):
        if(i==j):
            counter=counter+1
    accuracy_mnist=counter/len(calculated)
    
    print("\n|||||||||||||||||**SUPPORT VECTOR MACHINE**||||||||||||||||")
    print("----------------Accuracy for SVM on mnist test set---------------")
    print("accuracy :")
    print(accuracy_mnist)
    
    confusion_mat_mnist=confusion_matrix(y_test,calculated)
    print("confusion matrix")
    print(confusion_mat_mnist)
    
    
    calculated_usps=model.predict(x_testusps)
    counter_usps=0
    for i, j in zip(calculated_usps,y_testusps):
        if(i==j):
            counter_usps=counter_usps+1
    accuracy_usps=counter_usps/len(y_testusps)
    print("----------------Accuracy for SVM on USPS test set---------------")
    print("\n accuracy :")
    print(accuracy_usps)
    
    confusion_mat_usps=confusion_matrix(y_testusps,calculated_usps)
    print("confusion matrix")
    print(confusion_mat_usps)
    


# ## TESTING NEURAL NETWORK ON USPS DATA

# In[ ]:


def neural_test(x_test, y_test, model):
    
    y_test1=encodeLabel(y_test)
    loss, accuracy = model.evaluate(x_test,y_test1,batch_size=128,verbose=0)
    calculated=model.predict(x_test)
    
    e=np.argmax(calculated,axis=1)
    print("------TESTING STATISTICS ON USPS DATASET----------\n")
    print("\n accuracy for usps dataset:")
    print(accuracy)
    print("\nloss percentage for testing on usps dataset:", loss)
    confusion_mat_usps=confusion_matrix(y_test, e)
    print("confusion matrix :")
    print(confusion_mat_usps)


# ## Neural Network Classifier

# In[ ]:


def neural_network(x_train,y_train,x_test,y_test):
    
    input_size = x_train.shape[1]
    model=getmodel(input_size)
    validation_data_split = 0.2
    num_epochs = 1000
    model_batch_size = 300
    tb_batch_size = 32
    early_patience = 10
    tensorboard_cb   = TensorBoard(log_dir='logs', batch_size= tb_batch_size, write_graph= True)
    earlystopping_cb = EarlyStopping(monitor='val_loss', verbose=1, patience=early_patience, mode='min')
    
   
    y_train=encodeLabel(y_train)
    history = model.fit(x_train, y_train, validation_split=validation_data_split, epochs=num_epochs, batch_size=model_batch_size, callbacks = [tensorboard_cb,earlystopping_cb])
    y_test1= encodeLabel(y_test)

    print("\n ************NEURAL NETWORK************")
    loss_mnist, accuracy_mnist = model.evaluate(x_test,y_test1,batch_size=128,verbose=1)
    print("\n")
    print("---------TESTING STATISTICS ON MNIST DATASET-------------")
    print("\n accuarcy", accuracy_mnist)
    print("loss percentage", loss_mnist)
    
    calculated_mnist=model.predict(x_test)
    d=np.argmax(calculated_mnist,axis=1)
    confusion_matrix_mnist=confusion_matrix(y_test,d)
    print("confusion matrix :")
    print(confusion_matrix_mnist)
    return model



# In[ ]:



def encodeLabel(labels):
    return np_utils.to_categorical(np.array(labels),10)


def getmodel(input_size):
    
    drop_out = 0.15
    first_dense_layer_nodes  = 64
    second_dense_layer_nodes = 32
    third_dense_layer_nodes = 10
    model = Sequential()
    
    model.add(Dense(first_dense_layer_nodes, input_dim=input_size))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(second_dense_layer_nodes))
    model.add(Activation('sigmoid'))
    
    model.add(Dense(third_dense_layer_nodes))
    model.add(Activation('softmax'))
    

    
    
    opt = Adam()

    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model


# ## Random Forest

# In[ ]:


def random_forest(train_x,train_y,test_x,test_y,usps_x,usps_y):
    num_trees=450
    rf=RandomForestClassifier(n_estimators=num_trees)
    rf.fit(train_x,train_y)
    random_forest_test(test_x,test_y,rf,usps_x,usps_y)


# In[ ]:


def random_forest_test(testx,testy,model, USPS, usps_y):
    calculated=model.predict(testx)

    
    counter=0
    for i, j in zip(calculated,testy):
        if(i==j):
            counter=counter+1
    accuracy_mnist=counter/len(calculated)
    
    print("\n||||||||||||||||||||||**RANDOM FOREST**|||||||||||||||||||||||||||")
    print("----------------Accuracy for random forest on MNIST test set---------------")
    print("\n accuracy :",accuracy_mnist)
    
    confusion_mat_mnist=confusion_matrix(testy, calculated)
    print("\n confusion matrix ",confusion_mat_mnist)
    
    calculated_usps=model.predict(USPS)
    counter_usps=0
    for i, j in zip(calculated_usps,usps_y):
        if(i==j):
            counter_usps=counter_usps+1
    accuracy_usps=counter_usps/len(usps_y)
    
    print("--------------------------------------------------------------------------")
    print("----------------Accuracy for random forest on USPS test set---------------")
    print("\n accuracy :",accuracy_usps)
    confusion_mat_usps=confusion_matrix(usps_y,calculated_usps)
    print("confusion matrix")
    print(confusion_mat_usps)


# ## IMPORTS

# In[ ]:


if  __name__ == "__main__":
    from PIL import Image
    import os
    from sklearn.ensemble import RandomForestClassifier
    from matplotlib import pyplot as plt
    from sklearn.svm import SVC
    import pickle
    import gzip
    from sklearn.metrics import confusion_matrix
    from keras.layers import Dense,Activation, Dropout
    from keras.models import Sequential
    from keras.optimizers import Adam, RMSprop
    from keras.callbacks import EarlyStopping, TensorBoard
    from keras.utils import np_utils
    import numpy as np
    main()

