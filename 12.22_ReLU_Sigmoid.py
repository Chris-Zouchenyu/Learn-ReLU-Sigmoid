#非线性激活
import torch
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module
import torch.nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#搭建神经网络
class NN(Module):
    def __init__(self):
        super(NN,self).__init__()
        self.relu1 = ReLU(inplace=False)
        #inplace 的作用 是否替换原数据 如果为True 则替换原数据
        # input = -1                        input = -1
        # ReLU(input,inplace=True)          output = ReLU(input,inplace = False)
        # input = 0                         input = -1
        self.sigmoid1 = Sigmoid()
    def forward(self,input,module):
        if module == 'relu':
            output = self.relu1(input)
            return output
        if module == 'sigmoid':
            output = self.sigmoid1(input)
            return output
        
n = NN()

test_dataset = torchvision.datasets.CIFAR10(root='./dataset'
                                            ,transform = torchvision.transforms.ToTensor()
                                            ,train = False
                                            ,download = True)
test_loader = DataLoader(dataset = test_dataset
                         ,batch_size = 64#batch_size:从数据集中每次去四个打包
                         ,shuffle = True#shuffle:是否打乱
                         ,num_workers = 0#
                         ,drop_last = True)#按batch_size打包后是否舍弃余下的数据 比如100张按照每30张图片一包打包，会剩下来10张图片 若drop_last = False 就不舍弃


input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,-1,-1],
                      [-2,-1,0,-1,-1]],dtype=torch.float32)
input = torch.reshape(input,(-1,1,5,5))

writer = SummaryWriter('sigmoid & relu')
step = 0
for data in test_loader:
    imgs,targets = data
    output_sigmoid = n(input,'sigmoid')
    output_ReLU = n(input,'relu')
    writer.add_images('input_sigmoid',imgs,step,1)
    writer.add_images('output_sigmoid',output_sigmoid,step,2)
    writer.add_images('output_relu',output_ReLU,3)
    step+=1
writer.close()