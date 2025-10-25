import torch 
import torch.nn as nn   #这是神经网络模块
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["SimHei"]  # 设置字体，解决中文显示问题
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 定义模型
class TorchModel(nn.Module):
    def __init__(self):
        super(TorchModel,self).__init__()
        self.layer = nn.Linear(5,5)
        self.loss = nn.functional.cross_entropy
    
    def forward(self,x,y=None):
        y_pred = self.layer(x)
        if y is  None:
            return torch.softmax(y_pred,axis=-1)
        else:
            return self.loss(y_pred,y)
# 加载数据
def build_sample():
    x = np.random.rand(5)
    y = np.argmax(x)
    return x,y

def build_batch(sample_length):
    X,Y = [],[]
    for _ in range(sample_length):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

# 训练模型+评估模型+保存模型
def evaluate(model):
    model.eval()
    x,y = build_batch(100)
    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        for yp,yt in zip(y_pred,y):
            if torch.argmax(yp) == yt:
                correct += 1
            else:
                wrong +=1
    return correct/(correct+wrong)

def main():
    epoch_num = 20
    batch_size = 20
    train_sample_num = 10000
    learning_rate = 0.01
    model = TorchModel()
    optim = torch.optim.Adam(model.parameters(),lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(train_sample_num//batch_size):
            x,y = build_batch(batch_size)
            loss = model(x,y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(float(loss.item()))
        acc = evaluate(model)  
        log.append([acc,np.mean(watch_loss)])  
        print(f"epoch:{epoch},预测准确率：{acc},loss:{np.mean(watch_loss)}")  
    
    plt.title('训练过程的模型准确度和损失值变化')
    plt.plot(range(len(log)),[l[0] for l in log],label='acc')
    plt.plot(range(len(log)),[l[1] for l in log],label='loss')
    plt.legend()
    plt.show()
    torch.save(model.state_dict(),f'model/{datetime.now().strftime('%Y-%m-%d')}.bin')

def predict(model_path,input_vector):
    model = TorchModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(input_vector))
    for x,yp in zip(input_vector,y_pred):
        print(f"输入的向量：{x}，预测值：{yp}，预测类别：{torch.argmax(yp)}")
            
    
if __name__=='__main__':
    main()
    input_vector = [[1.3,1.4,0.1,0.5,9.1],
                    [5.2,4,1.5,7.4,2.5],
                    [9.1,3.6,90.1,10.8,1.6]]
    model_path = rf'model/{datetime.now().strftime('%Y-%m-%d')}.bin'
    predict(model_path,input_vector)
