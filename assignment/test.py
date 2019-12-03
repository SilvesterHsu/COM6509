%matplotlib inline
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
SCALE = 0.3
BATCH = 8

# %% Q1
class CIFAR10Dataset(torch.utils.data.Dataset):
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
        return image, label

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.clip(npimg,0,1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

classes = ('cat', 'dog')
data_source = {'train':True,'test':False}
data_process = ('raw','noise')

dataset = {k: {p: CIFAR10Dataset(classes, root='./data', train=v,
                    download=True, transform=transform[p]) for p in data_process} for k,v in data_source.items()}

loader = {k: {p: torch.utils.data.DataLoader(dataset[k][p], batch_size=BATCH,
                    shuffle=False, num_workers=2) for p in data_process} for k in data_source}

images, labels = next(iter(loader['train']['raw']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))

images, labels = next(iter(loader['train']['noise']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))

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
    imshow(torchvision.utils.make_grid(cat_tensor))
    imshow(torchvision.utils.make_grid(dog_tensor))

# %% Q2
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

def loadPCAData(data):
    x = data.normal_images.numpy().transpose(0,2,3,1).reshape(data.length,-1)
    y = np.array(data.labels)
    return x,y

class PCA(PCA):
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

    def plot_image(self,images,w,h,c):
        images_recovery = images.reshape(images.shape[0],w,h,c)[:5]
        scale = max(abs(images_recovery.max()),abs(images_recovery.min()))
        self.__plot_gallery(images_recovery/scale,n_row=1,n_col=5)

    def getSample_k(self,num = 7):
        variance_ratio = np.cumsum(self.explained_variance_ratio_)
        sellect_ratio = np.linspace(variance_ratio.min(),1,num+2)[1:-1]
        sellect_feature_index = [np.where(variance_ratio<=ratio)[0][-1] for ratio in sellect_ratio]
        return sellect_feature_index

class Multiclassifier():
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
        model = GaussianNB().fit(self.train_x, self.train_y)
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_x))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_x))
        test_pre_proba = model.predict_proba(self.test_x)

        print('Train Accurate:', train_accurate,'\n')
        print('Test Accurate:', test_accurate,'\n')

        if self.print_ROC:
            fpr,tpr,_ = roc_curve(self.test_y, test_pre_proba[:,1])
            roc_auc = auc(fpr, tpr)
            self.roc_dict = {'fpr':fpr,'tpr':tpr,'auc':roc_auc}
        return train_accurate,test_accurate

    def NBclassifier_raw(self):
        model = GaussianNB().fit(self.train_raw, self.train_y)
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_raw))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_raw))
        test_pre_proba = model.predict_proba(self.test_raw)

        print('Train Accurate:', train_accurate,'\n')
        print('Test Accurate:', test_accurate,'\n')

        if self.print_ROC:
            fpr,tpr,_ = roc_curve(self.test_y, test_pre_proba[:,1])
            roc_auc = auc(fpr, tpr)
            self.roc_dict = {'fpr':fpr,'tpr':tpr,'auc':roc_auc}
        return train_accurate,test_accurate

    def LRclassifier(self):
        model =LogisticRegression(solver='sag',multi_class='multinomial',tol=0.5).fit(self.train_x, self.train_y)
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_x))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_x))
        test_pre_proba = model.predict_proba(self.test_x)

        print('Train Accurate:', train_accurate,'\n')
        print('Test Accurate:', test_accurate,'\n')

        return train_accurate,test_accurate

    def LRclassifier_raw(self):
        model =LogisticRegression(solver='sag',multi_class='multinomial',tol=0.5).fit(self.train_raw, self.train_y)
        train_accurate = accuracy_score(self.train_y, model.predict(self.train_raw))
        test_accurate = accuracy_score(self.test_y, model.predict(self.test_raw))
        test_pre_proba = model.predict_proba(self.test_raw)

        print('Train Accurate:', train_accurate,'\n')
        print('Test Accurate:', test_accurate,'\n')

        return train_accurate,test_accurate

'''
def classifier(pca,train_x,train_y,test_x,test_y,k,reduce = True, ROC = True, LR = False):
    # Classifier
    if LR:
        model =LogisticRegression(solver='sag',multi_class='multinomial',tol=0.5)
        train_x_feature = pca.getFeature(train_x,k) if reduce else train_x
        model.fit(train_x_feature, train_y)
    else:
        model = GaussianNB()
        train_x_feature = pca.getFeature(train_x,k) if reduce else train_x
        model.fit(train_x_feature, train_y)

    # Train Acc
    train_pre = model.predict(train_x_feature)
    train_true = np.array(train_y)

    # Test Acc
    if reduce:
        print('Get features')
        test_x_feature = pca.getFeature(test_x,k)
    else:
        test_x_feature = test_x

    test_pre = model.predict(test_x_feature)
    test_pre_proba = model.predict_proba(test_x_feature)
    test_true = np.array(test_y)

    test_accurate = accuracy_score(test_true, test_pre)
    train_accurate = accuracy_score(train_true, train_pre)

    print('Train Accurate:', train_accurate,'\n')
    print('Test Accurate:', test_accurate,'\n')

    if ROC == True:
        # Evaluate
        def get_roc(y, y_proba, class_num = 1):
            fpr,tpr,_ = roc_curve(y, y_proba[:,class_num])
            roc_auc = auc(fpr, tpr)
            return fpr,tpr,roc_auc
        fpr,tpr,roc_auc = get_roc(test_true,test_pre_proba)
        return train_accurate, test_accurate, fpr, tpr, roc_auc

    return train_accurate, test_accurate
'''
# Part1: get training dataset
train_x,train_y = loadPCAData(dataset['train']['raw'])
test_x,test_y = loadPCAData(dataset['test']['raw'])

# Part2: PCA fit and get a set of k
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)
pca = pca.fit(train_x)
pca.plot_eigenfaces()
pca.plot_curve(np.cumsum(pca.explained_variance_ratio_),
            'Cumulative variance ratio of principal components',
            'Principle component number')
sellect_feature_index = pca.getSample_k(7)

# Part3: train with classifier
mode = Multiclassifier(train_x,train_y,test_x,test_y,print_ROC = True)
test_acc_list = list()
roc_dict = dict()
_,test_acc = mode.NBclassifier_raw()
test_acc_list.append(test_acc)
roc_dict['original'] = mode.roc_dict
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # image reconstruction
    train_x_recon = pca.getReconstruct(train_x,k)
    pca.plot_image(train_x, 32, 32, 3)
    pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc = mode.NBclassifier()
    test_acc_list.append(test_acc)
    roc_dict[k] = mode.roc_dict

# Part4: plot Accurate, ROC and AUC
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],test_acc_list)
plt.title('Accurate')
plt.show()
plot_roc(roc_dict)
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],[v['auc'] for k,v in roc_dict.items()])
plt.title('AUC')
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
_,test_acc = mode.NBclassifier_raw()
test_acc_list.append(test_acc)
roc_dict['original'] = mode.roc_dict
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # image reconstruction
    train_x_recon = pca.getReconstruct(train_x,k)
    pca.plot_image(train_x, 32, 32, 3)
    pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc = mode.NBclassifier()
    test_acc_list.append(test_acc)
    roc_dict[k] = mode.roc_dict

# Part4: plot Accurate, ROC and AUC
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],test_acc_list)
plt.title('Accurate')
plt.show()
plot_roc(roc_dict)
plt.bar(['original','k1','k2','k3','k4','k5','k6','k7'],[v['auc'] for k,v in roc_dict.items()])
plt.title('AUC')
plt.show()

# %% 10 classifier
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

images, labels = next(iter(loader['train']['raw']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))

images, labels = next(iter(loader['train']['noise']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))


# PCA
# Part1: get training dataset
train_x,train_y = loadPCAData(dataset['train']['raw'])
test_x,test_y = loadPCAData(dataset['test']['raw'])

# Part2: PCA fit and get a set of k
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)
pca = pca.fit(train_x)
sellect_feature_index = pca.getSample_k(3)

# Part3: train with classifier
mode = Multiclassifier(train_x,train_y,test_x,test_y,print_ROC = True)
test_acc_list = list()
_,test_acc = mode.LRclassifier_raw()
test_acc_list.append(test_acc)
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # image reconstruction
    train_x_recon = pca.getReconstruct(train_x,k)
    pca.plot_image(train_x, 32, 32, 3)
    pca.plot_image(train_x_recon, 32, 32, 3)
    # updata k
    mode.updateDataset(pca.getFeature(train_x,k),pca.getFeature(test_x,k))
    _,test_acc = mode.LRclassifier()
    test_acc_list.append(test_acc)

# Part4: plot Accurate, ROC and AUC
plt.bar(['original','k1','k2','k3'],test_acc_list)
plt.title('Accurate')
plt.show()
