%matplotlib inline
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression

torch.manual_seed(1720410)
IMAGE_SIZE = torch.Size([3,32,32])
NOISE = torch.rand(IMAGE_SIZE)
SCALE = 0.4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH = 8 if device == torch.device("cpu") else 1000

# %% Q1
class CIFAR10Dataset(torch.utils.data.Dataset):
    '''CIFAR10Dataset

    Custom data collection loading method. Extract data from selected labels only.
    For example:
        target_classes  = ('cat','dog') # Extract cat and dog data
        target_classes  = ('cat','dog','bird','ship') # Extract cat, dog, bird and ship data

    '''
    def __init__(self, target_classes, root='./data', train=True, download=True, transform=None):
        self.images = list()
        self.labels = list()
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.__loaddata(target_classes)
        self.normal_images = torch.clamp(self.NormalizeALL(),-1,1)
        self.length = self.__len__()
        assert len(self.images) == len(self.labels)

    def __loaddata(self, target_classes):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=self.train, download=self.download, transform=None)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        labels = np.array(dataset.targets)
        mask = np.zeros(len(labels)) == 1
        for item in target_classes:
            mask |= (labels == classes.index(item))
        mask = np.where(mask)
        self.labels = list(map(lambda x: target_classes.index(classes[x]),labels[mask]))
        self.images = dataset.data[mask]

    # Normalize all the dataset
    def NormalizeALL(self):
        normal = torch.zeros(self.images.shape[0],*IMAGE_SIZE)
        for i in range(self.images.shape[0]):
            normal[i] = self.transform(self.images[i])
        return normal

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.clamp(image,-1,1), label

# show images from pytorch loader
def imshow(img,title = None,size = None):
    img = img / 2 + 0.5
    npimg = img.numpy()
    if size:
        plt.figure(figsize=size)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)
    plt.show()

# basic parameters
classes = ('cat', 'dog')
data_source = {'train':True,'test':False}
data_process = ('raw','noise')

# set transform for dataset
transform = {
    'raw': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'noise': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + SCALE * NOISE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])}
# load 4 different datasets:
# train with raw data, train with noise data
# test with raw data, test with noise data
CatDog = dataset = {k: {p: CIFAR10Dataset(classes, root='./data', train=v,
                    download=True, transform=transform[p]) for p in data_process} for k,v in data_source.items()}
# define loader to get batch for deep network training
loader = {k: {p: torch.utils.data.DataLoader(dataset[k][p], batch_size=BATCH,
                    shuffle=False, num_workers=2) for p in data_process} for k in data_source}
# Show 10 pairs of original and noisy images of cats
# and 10 pairs of original and noisy images of dogs
for p in data_process:
    c,cat_tensor = 0,torch.zeros([10,3,32,32])
    d,dog_tensor = 0,torch.zeros([10,3,32,32])
    for image,label in dataset['train'][p]:
        if classes[label] == 'cat' and c<10:
            cat_tensor[c] = image
            c += 1
        if classes[label] == 'dog' and d<10:
            dog_tensor[d] = image
            d += 1
    print(p)
    imshow(torchvision.utils.make_grid(cat_tensor,nrow=5),'Cats ({})'.format(p))
    imshow(torchvision.utils.make_grid(dog_tensor,nrow=5),'Dogs ({})'.format(p))

# %% Q2
# plot roc data
def plot_roc(roc_dict):
    plt.title('ROC')
    i = 0
    for k,v in roc_dict.items():
        fpr,tpr,auc = v['fpr'],v['tpr'],v['auc']
        if k == 'original':
            plt.plot(fpr, tpr,label='original, AUC = %0.4f'% auc)
        else:
            i+=1
            plt.plot(fpr, tpr,label='k'+str(i)+', AUC = %0.4f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

# Separate the input and labels from the dataset and flatten the input
def loadPCAData(data):
    x = data.normal_images.numpy().transpose(0,2,3,1).reshape(data.length,-1)
    y = np.array(data.labels)
    return x,y

class PCA(PCA):
    '''PCA
    Redefining the PCA class, inherited from the parent PCA.
    Add several problem-specific functions to PCA.
    '''
    def plot_eigenfaces(self,w=32,h=32,c=3):
        eigenfaces = self.components_.reshape(-1,w,h,c)
        eigenface_titles = ["feature %d" %
                                  i for i in range(eigenfaces.shape[0])]
        self.__plot_gallery(eigenfaces,eigenface_titles,eigen = True)

    def __plot_gallery(self, images, titles=None, n_row=3, n_col=4, eigen = False):
        """Helper function to plot a gallery of portraits"""
        plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
        plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(n_row * n_col):
            plt.subplot(n_row, n_col, i + 1)
            if eigen == True:
                plt.imshow(np.clip(images[i]*20,0,1))
            else:
                plt.imshow((images[i]+1)/2)
            if titles:
                plt.title(titles[i], size=12)
            plt.xticks(())
            plt.yticks(())
        plt.show()

    def plot_curve(self, data, title = None, xlabel = None, ylabel = None):
        plt.plot(range(1, data.shape[0] + 1), data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def getFeature(self,data,k):
        feature_k = self.components_[:k]
        feature_data = data.dot(feature_k.T)
        return feature_data

    def getReconstruct(self,data,k):
        feature_data = self.getFeature(data,k)
        reconstruct_data = feature_data.dot(self.components_[:k])
        return reconstruct_data

    def plot_image(self,images,w,h,c,title = None):
        images_recovery = images.reshape(images.shape[0],w,h,c)[:5]
        scale = max(abs(images_recovery.max()),abs(images_recovery.min()))
        self.__plot_gallery(images_recovery/scale,n_row=1,n_col=5)

    def getSample_k(self,num = 7):
        variance_ratio = np.cumsum(self.explained_variance_ratio_)
        sellect_ratio = np.linspace(variance_ratio.min(),1,num+2)[1:-1]
        sellect_feature_index = [np.where(variance_ratio<=ratio)[0][-1] for ratio in sellect_ratio]
        return sellect_feature_index

class Multiclassifier():
    '''Multiclassifier
    A collection of classifiers for easy invocation
    '''
    def __init__(self,train_raw,train_y,test_raw,test_y,print_ROC = False):
        self.print_ROC = print_ROC
        self.train_raw = train_raw
        self.train_y = train_y
        self.test_raw = test_raw
        self.test_y = test_y

    def updateDataset(self,train_x,test_x):
        self.train_x = train_x
        self.test_x = test_x

    def NBclassifier(self):
        tic = time.time()
        model = GaussianNB().fit(self.train_x, self.train_y)
        self.train_time = time.time() - tic
        tic = time.time()
        _ = model.predict(self.test_x)
        self.test_time = time.time() - tic
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_x))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_x))
        test_pre_proba = model.predict_proba(self.test_x)

        print('Train Accurate:', train_accurate)
        print('Test Accurate:', test_accurate,'\n')

        if self.print_ROC:
            fpr,tpr,_ = roc_curve(self.test_y, test_pre_proba[:,1])
            roc_auc = auc(fpr, tpr)
            self.roc_dict = {'fpr':fpr,'tpr':tpr,'auc':roc_auc}

        confusion = [model.predict(self.test_x),self.test_y]
        return train_accurate,test_accurate,confusion

    def NBclassifier_raw(self):
        tic = time.time()
        model = GaussianNB().fit(self.train_raw, self.train_y)
        self.train_time = time.time() - tic
        tic = time.time()
        _ = model.predict(self.test_raw)
        self.test_time = time.time() - tic
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_raw))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_raw))
        test_pre_proba = model.predict_proba(self.test_raw)

        print('Train Accurate:', train_accurate)
        print('Test Accurate:', test_accurate,'\n')

        if self.print_ROC:
            fpr,tpr,_ = roc_curve(self.test_y, test_pre_proba[:,1])
            roc_auc = auc(fpr, tpr)
            self.roc_dict = {'fpr':fpr,'tpr':tpr,'auc':roc_auc}
        confusion = [model.predict(self.test_raw), self.test_y]
        return train_accurate,test_accurate,confusion

    def LRclassifier(self):
        tic = time.time()
        model =LogisticRegression(solver='sag',multi_class='multinomial',tol=0.5).fit(self.train_x, self.train_y)
        self.train_time = time.time() - tic
        tic = time.time()
        _ = model.predict(self.test_x)
        self.test_time = time.time() - tic
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_x))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_x))
        test_pre_proba = model.predict_proba(self.test_x)

        print('Train Accurate:', train_accurate)
        print('Test Accurate:', test_accurate,'\n')

        confusion = [model.predict(self.test_x),self.test_y]
        return train_accurate,test_accurate,confusion

    def LRclassifier_raw(self):
        tic = time.time()
        model =LogisticRegression(solver='sag',multi_class='multinomial',tol=0.5).fit(self.train_raw, self.train_y)
        self.train_time = time.time() - tic
        tic = time.time()
        _ = model.predict(self.test_raw)
        self.test_time = time.time() - tic
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_raw))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_raw))
        test_pre_proba = model.predict_proba(self.test_raw)

        print('Train Accurate:', train_accurate)
        print('Test Accurate:', test_accurate,'\n')

        confusion = [model.predict(self.test_raw), self.test_y]
        return train_accurate,test_accurate,confusion

# Part1: get training dataset
train_x,train_y = loadPCAData(dataset['train']['raw'])
test_x,test_y = loadPCAData(dataset['test']['raw'])

# Part2: PCA fit and get a set of k choices
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)
pca = pca.fit(train_x)
#pca.plot_eigenfaces()
pca.plot_curve(np.cumsum(pca.explained_variance_ratio_),
            'Cumulative variance ratio of principal components',
            'Principle component number')
sellect_feature_index = pca.getSample_k(7)

# Part3: train with classifier
mode = Multiclassifier(train_x,train_y,test_x,test_y,print_ROC = True)
test_acc_list = list()
roc_dict = dict()
print('Raw')
_,test_acc,_ = mode.NBclassifier_raw()
test_acc_list.append(test_acc)
roc_dict['original'] = mode.roc_dict
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # show image reconstruction
    #train_x_recon = pca.getReconstruct(train_x,k)
    #pca.plot_image(train_x, 32, 32, 3)
    #pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc,_ = mode.NBclassifier()
    test_acc_list.append(test_acc)
    roc_dict[k] = mode.roc_dict

# Part4: plot Accurate, ROC and AUC
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],test_acc_list)
plt.title('Accurate')
plt.show()
plot_roc(roc_dict)
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],[v['auc'] for k,v in roc_dict.items()])
plt.title('Area under the ROC curve')
plt.show()

# %% Noise
# Part1: get training dataset
train_x,train_y = loadPCAData(dataset['train']['noise'])
test_x,test_y = loadPCAData(dataset['test']['noise'])

# Part2: PCA fit and get a set of k
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)
pca = pca.fit(train_x)

# Part3: train with classifier
mode = Multiclassifier(train_x,train_y,test_x,test_y,print_ROC = True)
test_acc_list = list()
roc_dict = dict()
_,test_acc,_ = mode.NBclassifier_raw()
test_acc_list.append(test_acc)
roc_dict['original'] = mode.roc_dict
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # image reconstruction
    #train_x_recon = pca.getReconstruct(train_x,k)
    #pca.plot_image(train_x, 32, 32, 3)
    #pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc,_ = mode.NBclassifier()
    test_acc_list.append(test_acc)
    roc_dict[k] = mode.roc_dict

# Part4: plot Accurate, ROC and AUC
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],test_acc_list)
plt.title('Accurate')
plt.show()
plot_roc(roc_dict)
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],[v['auc'] for k,v in roc_dict.items()])
plt.title('Area under the ROC curve')
plt.show()

# %% 10 classifier
# set transform for dataset
transform = {
    'raw': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'noise': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + SCALE * NOISE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_source = {'train':True,'test':False}
data_process = ('raw','noise')

dataset = {k: {p: CIFAR10Dataset(classes, root='./data', train=v,
                    download=True, transform=transform[p]) for p in data_process} for k,v in data_source.items()}

loader = {k: {p: torch.utils.data.DataLoader(dataset[k][p], batch_size=BATCH,
                    shuffle=False, num_workers=2) for p in data_process} for k in data_source}
'''
images, labels = next(iter(loader['train']['raw']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))

images, labels = next(iter(loader['train']['noise']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))
'''
confusion_matrix_collect = list()
accuracy_collect = list()
train_time_collect = list()
test_time_collect = list()
#%% 4 Naive Bayes
# Part1: get training dataset
train_x,train_y = loadPCAData(dataset['train']['raw'])
test_x,test_y = loadPCAData(dataset['test']['raw'])

# Part2: PCA fit and get a set of k
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)
pca = pca.fit(train_x)
sellect_feature_index = pca.getSample_k(3)

# Part3: train with classifier
mode = Multiclassifier(train_x,train_y,test_x,test_y)
test_acc_list = list()

_,test_acc,confusion = mode.NBclassifier_raw()
test_acc_list.append(test_acc)
# collect data
confusion_matrix_collect.append(confusion)
accuracy_collect.append(test_acc)
train_time_collect.append(mode.train_time)
test_time_collect.append(mode.test_time)
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # image reconstruction
    #train_x_recon = pca.getReconstruct(train_x,k)
    #pca.plot_image(train_x, 32, 32, 3)
    #pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc,confusion = mode.NBclassifier()
    test_acc_list.append(test_acc)
    # collect data
    confusion_matrix_collect.append(confusion)
    accuracy_collect.append(test_acc)
    train_time_collect.append(mode.train_time)
    test_time_collect.append(mode.test_time)

# Part4: plot Accurate, ROC and AUC
#plt.bar(['original','k1','k2','k3'],test_acc_list)
#plt.title('4 Naive Bayes Accurate')
#plt.show()

#%% 4 Logistic Regression
# PCA
# Part1: get training dataset
train_x,train_y = loadPCAData(dataset['train']['raw'])
test_x,test_y = loadPCAData(dataset['test']['raw'])

# Part2: PCA fit and get a set of k
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)
pca = pca.fit(train_x)
#sellect_feature_index = pca.getSample_k(3)

# Part3: train with classifier
mode = Multiclassifier(train_x,train_y,test_x,test_y)
test_acc_list = list()
_,test_acc,confusion = mode.LRclassifier_raw()
test_acc_list.append(test_acc)
# collect data
confusion_matrix_collect.append(confusion)
accuracy_collect.append(test_acc)
train_time_collect.append(mode.train_time)
test_time_collect.append(mode.test_time)
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # image reconstruction
    #train_x_recon = pca.getReconstruct(train_x,k)
    #pca.plot_image(train_x, 32, 32, 3)
    #pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc,confusion = mode.LRclassifier()
    test_acc_list.append(test_acc)
    # collect data
    confusion_matrix_collect.append(confusion)
    accuracy_collect.append(test_acc)
    train_time_collect.append(mode.train_time)
    test_time_collect.append(mode.test_time)

# Part4: plot Accurate, ROC and AUC
#plt.bar(['original','k1','k2','k3'],test_acc_list)
#plt.title('4 Logistic Regression Accurate')
#plt.show()

# %% 1 CNN
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainloader = loader['train']['raw']
testloader = loader['test']['raw']

net = Net()
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

tic = time.time()
for epoch in range(3):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
train_time_collect.append(time.time() - tic)
correct = 0
total = 0
confusion = [torch.zeros(len(testloader),BATCH) for i in range(2)]
i = 0
tic = time.time()
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        confusion[0][i,:] = predicted.view(-1)
        confusion[1][i,:] = labels.view(-1)
        i += 1
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
test_time_collect.append(time.time()-tic)
print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
confusion = [confusion[i].view(-1).numpy() for i in range(2)]
confusion_matrix_collect.append(confusion)
accuracy_collect.append(correct / total)
# %% plot the classification accuracy, total training time, and total test time
catagory = ['NB','LR']
title = ['raw','k1','k2','k3']
titles = [c+' '+t for c in catagory for t in title] + ['CNN']
plt.figure(figsize=(10, 5))
plt.bar(titles,accuracy_collect)
plt.title("Classification Accuracy",size = 18)
plt.show()
plt.figure(figsize=(10, 5))
plt.bar(titles,train_time_collect)
plt.title("Training Time",size = 18)
plt.show()
plt.figure(figsize=(10, 5))
plt.bar(titles,test_time_collect)
plt.title("Testing Time",size = 18)
plt.show()
# %% confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

catagory = ['Naive Bayes','Logistic Regression']
title = ['raw','k1','k2','k3']
titles = [c+' '+t for c in catagory for t in title] + ['CNN']
n_row = n_col = 3
#plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
plt.figure(figsize=(5 * n_col, 5 * n_row))
plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.2)
for i in range(n_row * n_col):
    plt.subplot(n_row, n_col, i + 1)
    mat = confusion_matrix(*confusion_matrix_collect[i])
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels='', yticklabels='')
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.title(titles[i], size=24)
    plt.xticks(())
    plt.yticks(())
plt.show()
# %% Q4
# set transform for dataset
transform = {
    'raw': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
    'noise': transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + SCALE * NOISE),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_source = {'train':True,'test':False}
data_process = ('raw','noise')

# load 4 different datasets:
# train with raw data, train with noise data
# test with raw data, test with noise data
dataset = {k: {p: CIFAR10Dataset(classes, root='./data', train=v,
                    download=True, transform=transform[p]) for p in data_process} for k,v in data_source.items()}

loader = {k: {p: torch.utils.data.DataLoader(dataset[k][p], batch_size=BATCH,
                    shuffle=False, num_workers=2) for p in data_process} for k in data_source}

# %%
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 1 input image channel, 16 output channel, 3x3 square convolution
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 7)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()  #to range [0, 1]
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


trainloader_noise = loader['train']['noise']
trainloader_raw = loader['train']['raw']

net = Autoencoder()
net.to(device)

def trainNet(net,trainloader_raw,trainloader_noise,lr = 0.001, max_epoch = 5):
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(max_epoch):
        running_loss = 0.0
        data_iter = iter(trainloader_raw)
        for i,data_noise in enumerate(trainloader_noise,0):
            data_raw = next(data_iter)
            inputs_raw = data_raw[0].to(device)
            inputs_noise = data_noise[0].to(device)
            optimizer.zero_grad()
            recon = net(inputs_noise)
            loss = criterion(recon, inputs_raw)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    print('Finished Training')

trainNet(net,trainloader_raw,trainloader_noise,lr = 0.001,max_epoch = 5)

testloader_noise = loader['test']['noise']
testloader_raw = loader['test']['raw']


# collect loss
loss = torch.zeros(len(testloader_noise),BATCH)
with torch.no_grad():
    criterion = nn.MSELoss()
    data_iter = iter(testloader_raw)
    for i,data_noise in enumerate(testloader_noise,0):
        data_raw = next(data_iter)
        inputs_raw = data_raw[0].to(device)
        inputs_noise = data_noise[0].to(device)
        recon = net(inputs_noise)
        for j in range(len(inputs)):
            loss[i,j] = criterion(inputs_noise[j],recon[j])

top_list = loss.view(-1).topk(30)[-1]
worst_data = torch.zeros(30*2,*IMAGE_SIZE)
l = 0
for index in top_list:
    worst_data[l] = dataset['test']['noise'].normal_images[index]
    worst_data[l+1] = dataset['test']['raw'].normal_images[index]
    l += 2
imshow(torchvision.utils.make_grid(worst_data,nrow=4),title = 'Worst Data',size = (10,30))

# %%
def MSE(net,testloader_raw,testloader_noise):
    loss = 0
    with torch.no_grad():
        data_iter = iter(testloader_raw)
        for i,data_noise in enumerate(testloader_noise,0):
            data_raw = next(data_iter)
            inputs_raw = data_raw[0].to(device)
            inputs_noise = data_noise[0].to(device)
            recon = net(inputs_noise)
            criterion = nn.MSELoss()
            loss += criterion(recon, inputs_raw)
    return loss/len(testloader_noise)

def hyperparametersTest(lr,max_epoch):
    trainloader_noise = loader['train']['noise']
    trainloader_raw = loader['train']['raw']
    net = Autoencoder()
    net.to(device)
    trainNet(net,trainloader_raw,trainloader_noise,lr = lr,max_epoch = max_epoch)
    testloader_noise = loader['test']['noise']
    testloader_raw = loader['test']['raw']
    MSE_loss = MSE(net,testloader_raw,testloader_noise)
    return MSE_loss.item()

print("lr as hyperparameters")
lr = [0.0001, 0.5, 1]
MSE_lr = [hyperparametersTest(lr = p,max_epoch = 2) for p in lr]
print("\n max_epoch as hyperparameters")
max_epoch = [2,4,6]
MSE_epoch = [hyperparametersTest(lr = 0.01,max_epoch = e) for e in max_epoch]

plt.bar(lr,MSE_lr,width=0.2)
plt.title("MSE under different learning rate")
plt.show()

plt.bar(max_epoch,MSE_epoch)
plt.title("MSE under different epoch")
plt.show()
