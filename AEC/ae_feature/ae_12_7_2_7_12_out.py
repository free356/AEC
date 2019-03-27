import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model #泛型模型
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn import preprocessing
import matplotlib.pyplot as plt


#写一个LossHistory类，保存loss和acc
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        """
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        """
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

data=pd.read_excel("./ae_input.xlsx",usecols=[2,3,4,5,6,7,8,9,10,11,12,13],header=None,names=['arr1','arr2','arr3','arr4','arr5','arr6','arr7','arr8','arr9','arr10','arr11','arr12'])
print(data.head())
print(type(data))
 
# 数据预处理
X_scaled = preprocessing.scale(data)


 
# 压缩特征维度至2维
encoding_dim = 2
 
# this is our input placeholder
input_img = Input(shape=(12,))
 
# 编码
encoded = Dense(7, activation='relu')(input_img)

encoder_output = Dense(2)(encoded)
 
# 解码层
decoded = Dense(7, activation='relu')(encoder_output)

decoded = Dense(12, activation='tanh')(decoded)
 
# 构建自编码模型
autoencoder = Model(inputs=input_img, outputs=decoded)
# 构建编码模型
encoder = Model(inputs=input_img, outputs=encoder_output)
 
autoencoder.summary()

# compile autoencoder
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])

history = LossHistory()
# training
autoencoder.fit(X_scaled, X_scaled, epochs=200, batch_size=32, shuffle=True,callbacks=[history])

#绘制acc-loss曲线
history.loss_plot('epoch')
encoder.predict(X_scaled)

