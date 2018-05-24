import os
import torch
import torchvision
import torchvision.datasets as dset
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from skimage import io, transform
import argparse

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--save', '-s', help='train and save', action="store_true")
group.add_argument('--load', '-l', help='load and test', action="store_true")

parser.add_argument('filename', help='name of the file to load or save')
parser.add_argument('--augmentation', '-a', help='data augmentation', action='store_true')
args = parser.parse_args()

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()
    
class siameseDataset(Dataset):
    
    def __init__(self,imageFolderDataset, txt_file, transform=None):
        self.imageFolderDataset = imageFolderDataset  
        self.lines = open(txt_file).readlines()
        self.transform = transform
        
    def __getitem__(self,index):
        line = self.lines[index].split(' ')
        path1 = self.imageFolderDataset.root + '/' + line[0]
        path2 = self.imageFolderDataset.root + '/' + line[1]
        result = int(line[2][0: -1])
        img1 = io.imread(path1)
        img2 = io.imread(path2)
        img1 = transform.resize(img1, (128, 128), preserve_range=True, mode='reflect')
        img2 = transform.resize(img2, (128, 128), preserve_range=True, mode='reflect')
        label = torch.from_numpy(np.array([result], dtype=np.float32))
        if args.augmentation:
            prob1 = np.random.uniform(0, 1)
            prob2 = np.random.uniform(0, 1)
            options = ['mirror', 'rotation', 'translation', 'scaling']
            inds = [0, 1, 2, 3]
            ranges = [(0, 1), (-30, 30), (-10, 10), (0.7, 1.3)]
            if prob1 >= 1 - 0.7:
                number = np.random.randint(1, 5, size=1)[0]
                tinds = np.random.choice(inds, number, replace=False)
                for tind in tinds:
                    if options[tind] == 'mirror':
                        axis = np.random.choice(ranges[tind], 1)[0]
                        if axis == 0:
                            img1 = img1[::-1, :, :]
                        else:
                            img1 = img1[:, ::-1, :]
                    elif options[tind] == 'rotation':
                        img1 = transform.rotate(img1, np.random.uniform(ranges[tind][0], ranges[tind][1]))
                    elif options[tind] == 'translation':
                        transxy = (np.random.uniform(ranges[tind][0], ranges[tind][1]), 
                                     np.random.uniform(ranges[tind][0], ranges[tind][1]))
                        tform = transform.AffineTransform(translation=transxy)
                        img1 = transform.warp(img1, tform)
                    elif options[tind] == 'scaling':
                        factors = (np.random.uniform(ranges[tind][0], ranges[tind][1]), 
                                     np.random.uniform(ranges[tind][0], ranges[tind][1]))
                        tform = transform.AffineTransform(scale=factors)
                        img1 = transform.warp(img1, tform)
            if prob2 >= 1 - 0.7:
                number = np.random.randint(1, 5, size=1)[0]
                tinds = np.random.choice(inds, number, replace=False)
                for tind in tinds:
                    if options[tind] == 'mirror':
                        axis = np.random.choice(ranges[tind], 1)[0]
                        if axis == 0:
                            img2 = img2[::-1, :, :]
                        else:
                            img2 = img2[:, ::-1, :]
                    elif options[tind] == 'rotation':
                        img2 = transform.rotate(img2, np.random.uniform(ranges[tind][0], ranges[tind][1]))
                    elif options[tind] == 'translation':
                        transxy = (np.random.uniform(ranges[tind][0], ranges[tind][1]), 
                                     np.random.uniform(ranges[tind][0], ranges[tind][1]))
                        tform = transform.AffineTransform(translation=transxy)
                        img2 = transform.warp(img2, tform)
                    elif options[tind] == 'scaling':
                        factors = (np.random.uniform(ranges[tind][0], ranges[tind][1]), 
                                     np.random.uniform(ranges[tind][0], ranges[tind][1]))
                        tform = transform.AffineTransform(scale=factors)
                        img2 = transform.warp(img2, tform)
        img1 = np.array(img1, dtype='uint8')
        img2 = np.array(img2, dtype='uint8')
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2 , label
    
    def __len__(self):
        return len(self.lines) - 1
    
class siamese(nn.Module):
    def __init__(self):
        super(siamese, self).__init__()
        self.cnn1 = nn.Sequential(
                  nn.Conv2d(3, 64, (5, 5), stride=(1, 1), padding=2),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(64),
                  nn.MaxPool2d(2, stride=(2, 2)),
                  
                  nn.Conv2d(64, 128, (5, 5), stride=(1, 1), padding=2),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(128),
                  nn.MaxPool2d(2, stride=(2, 2)),
                  
                  nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=1),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(256),
                  nn.MaxPool2d(2, stride=(2, 2)),
                  
                  nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=1),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(512),
                  )
        self.cnn2 = nn.Sequential(
                  nn.Linear(131072, 1024),
                  nn.ReLU(inplace=True),
                  nn.BatchNorm2d(1024),
                  )
        self.out_layer = nn.Linear(2048, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward_once(self, x):
        out = self.cnn1(x)
        out = out.view(out.size()[0], -1)
        out = self.cnn2(out)
        return out
        
    def forward(self, image1, image2):
        output1 = self.forward_once(image1)
        output2 = self.forward_once(image2)
        
        output = torch.cat((output1, output2), 1)
        output = self.out_layer(output)
        output = self.sigmoid(output)
        
        return output


trans = transforms.Compose([transforms.ToTensor()])
model = siamese().cuda()
dataset_folder = dset.ImageFolder(root='./lfw')
batchsize = 32

if args.save:
    train_file = 'train.txt'
    dataset = siameseDataset(imageFolderDataset=dataset_folder, txt_file=train_file, transform=trans)
    if args.augmentation:
        epochs = 100
    else:
        epochs = 20
    dataset_loader = DataLoader(dataset, batch_size=batchsize, shuffle=True, num_workers=8)
    dataiter = iter(dataset_loader)
    optimizer = optim.Adam(model.parameters(),lr=0.000001)
    lossfn = torch.nn.BCELoss().cuda()
    loss_log = []
    for epoch in range(epochs):
        for data in dataset_loader:
            image1, image2, label = data
        #    images = torch.cat((image1, image2), 0)
        #    grid = torchvision.utils.make_grid(images)
        #    imshow(grid)
            image1, image2, label = Variable(image1).cuda(), Variable(image2).cuda() , Variable(label).cuda()
            output = model(image1, image2)
            optimizer.zero_grad()
            loss = lossfn(output, label)
            loss.backward()
            optimizer.step()
            loss_log.append(loss.data[0])
        print 'epoch = ', epoch, ', loss = ', loss.data[0]
        
    torch.save(model.state_dict(), args.filename)
    plt.plot(range(len(loss_log)), loss_log)
    if not args.augmentation:
        plt.savefig('loss_history1.png')
    else:
        plt.savefig('loss_history_with_aug1.png')
    
if args.load:
    N = batchsize
    model.load_state_dict(torch.load(args.filename))
    model.eval()
    test_file = 'test.txt'
    testset = siameseDataset(imageFolderDataset=dataset_folder, txt_file=test_file, transform=trans)
    testset_loader = DataLoader(testset, batch_size=N, shuffle=True, num_workers=8)
    test_correct = 0
    count = 0
    for data in testset_loader:
        count += N
        image1, image2, label = data
        image1, image2, label = Variable(image1).cuda(), Variable(image2).cuda() , Variable(label).cuda()
        output = model(image1, image2)
        result = torch.round(output)
        test_correct += np.count_nonzero(label.cpu().data.numpy() == result.cpu().data.numpy())
            
    print "Accuracy on test set is ", (test_correct / float(count))
    
    train_file = 'train.txt'
    trainset = siameseDataset(imageFolderDataset=dataset_folder, txt_file=train_file, transform=trans)
    trainset_loader = DataLoader(trainset, batch_size=N, shuffle=True, num_workers=8)
    train_correct = 0 
    count = 0      
    for data in trainset_loader:
        count += N
        image1, image2, label = data
        image1, image2, label = Variable(image1).cuda(), Variable(image2).cuda() , Variable(label).cuda()
        output = model(image1, image2)
        result = torch.round(output)
        train_correct += np.count_nonzero(label.cpu().data.numpy() == result.cpu().data.numpy())
    
    print "Accuracy on train set is ",(train_correct / float(count))
    
#torch.save(model.state_dict(), './model.pt')
    
#train = dset.ImageFolder('./lfw', trans)
#test = dset.ImageFolder('./lfw', trans)
#N = 10
#train_loader = DataLoader(dataset=train, batch_size=N, shuffle=False, num_workers=2)
#test_loader = DataLoader(dataset=test, batch_size=N, shuffle=False, num_workers=2)
#dataiter = iter(train_loader)
#images, labels = dataiter.next()
#grid = torchvision.utils.make_grid(images)
#imshow(grid)
#print " ".join(["%s" % i for i in labels])
