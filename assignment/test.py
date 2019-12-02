%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

torch.manual_seed(1720410)
IMAGE_SIZE = torch.Size([3,32,32])
NOISE = torch.rand(IMAGE_SIZE)
SCALE = 0.45
BATCH = 8

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, target_classes, root='./data', train=True, download=True, transform=None):
        self.images = list()
        self.labels = list()
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.__loaddata(target_classes)
        self.normal_images = self.NormalizeALL()
        assert len(self.images) == len(self.labels)

    def __loaddata(self, target_classes):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=self.train, download=self.download, transform=None)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        import numpy as np
        labels = np.array(dataset.targets)
        mask = np.zeros(len(labels)) == 1
        for item in target_classes:
            mask |= (labels == classes.index(item))
        mask = np.where(mask)
        self.labels = list(map(lambda x: target_classes.index(classes[x]),labels[mask]))
        self.images = dataset.data[mask]

    def __loaddata_v1(self, target_classes):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=self.train, download=self.download, transform=None)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        for image, label in dataset:
            if classes[label] in target_classes:
                self.images.append(image)
                self.labels.append(target_classes.index(classes[label]))

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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg = np.clip(npimg,0,1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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

# %% PCA
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

def classifier(pca,train_x,train_y,test_x,test_y,k):
    # Classifier
    # print('Train with Naive Bayes Classifier')
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import roc_curve, auc
    model = GaussianNB()
    train_x_feature = pca.getFeature(train_x,k)
    model.fit(train_x_feature, train_y)

    # Train Acc
    train_pre = model.predict(train_x_feature)
    train_true = np.array(train_y)

    loss = (train_pre-train_true).dot((train_pre-train_true))
    train_accurate = 1 - loss/train_true.shape[0]
    print('Train Accurate:', train_accurate)

    # Test Acc
    test_x_feature = pca.getFeature(test_x,k)

    test_pre = model.predict(test_x_feature)
    test_pre_proba = model.predict_proba(test_x_feature)
    test_true = np.array(test_y)

    def get_roc(y, y_proba, class_num = 1):
        fpr,tpr,_=roc_curve(y, y_proba[:,class_num])
        roc_auc = auc(fpr, tpr)
        return fpr,tpr,roc_auc

    fpr,tpr,roc_auc = get_roc(test_true,test_pre_proba)

    loss = (test_pre-test_true).dot((test_pre-test_true))
    test_accurate = 1 - loss/test_true.shape[0]
    print('Test Accurate:', test_accurate,'\n')
    return train_accurate, test_accurate, fpr, tpr, roc_auc

def plot_roc(roc_dict):
    plt.title('ROC')
    i = 0
    for k,v in roc_dict.items():
        i+=1
        fpr,tpr,auc = v['fpr'],v['tpr'],v['auc']
        plt.plot(fpr, tpr,label='k'+str(i)+' AUC = %0.4f'% auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.show()

train_x = dataset['train']['raw'].normal_images.numpy().transpose(0,2,3,1)
train_x = train_x.reshape(train_x.shape[0],-1)
train_y = dataset['train']['raw'].labels
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)

pca = pca.fit(train_x)
pca.plot_eigenfaces()
variance_ratio = np.cumsum(pca.explained_variance_ratio_)
pca.plot_curve(variance_ratio,
            'Cumulative variance ratio of principal components',
            'Principle component number')

sellect_ratio = np.linspace(variance_ratio.min(),1,7+2)[1:-1]
sellect_feature_index = [np.where(variance_ratio<=ratio)[0][-1] for ratio in sellect_ratio]
sellect_feature_index

test_acc_list = list()
roc_dict = dict()
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # load train images
    train_x = dataset['train']['raw'].normal_images.numpy().transpose(0,2,3,1)
    train_x = train_x.reshape(train_x.shape[0],-1)
    train_y = dataset['train']['raw'].labels
    # reconstruct train images
    train_x_recon = pca.getReconstruct(train_x,k)
    pca.plot_image(train_x, 32, 32, 3)
    pca.plot_image(train_x_recon, 32, 32, 3)
    # load test images
    test_x = dataset['test']['raw'].normal_images.numpy().transpose(0,2,3,1)
    test_x = test_x.reshape(test_x.shape[0],-1)
    test_y = dataset['test']['raw'].labels

    _,test_acc,fpr,tpr,auc = classifier(pca,train_x,train_y,test_x,test_y,k)
    test_acc_list.append(test_acc)
    roc_dict[k] = {'fpr':fpr,'tpr':tpr,'auc':auc}

plt.bar(range(1,8),test_acc_list)
plt.show()
plot_roc(roc_dict)

# %% Test k from 1 to 300
test_acc_list = list()
for i in range(300):
    k = i+1
    train_x = dataset['train']['raw'].normal_images.numpy().transpose(0,2,3,1)
    train_x = train_x.reshape(train_x.shape[0],-1)
    train_y = dataset['train']['raw'].labels

    test_x = dataset['test']['raw'].normal_images.numpy().transpose(0,2,3,1)
    test_x = test_x.reshape(test_x.shape[0],-1)
    test_y = dataset['test']['raw'].labels
    print('K:',k)
    _,test_acc,_,_,_ = classifier(pca,train_x,train_y,test_x,test_y,k)
    test_acc_list.append(test_acc)
plt.figure(figsize=(12, 6))
plt.plot(range(300),test_acc_list)
# %% Noise
train_x = dataset['train']['noise'].normal_images.numpy().transpose(0,2,3,1)
train_x = train_x.reshape(train_x.shape[0],-1)
train_y = dataset['train']['noise'].labels
pca = PCA(n_components = 500, svd_solver='randomized', whiten=True)

pca = pca.fit(train_x)
pca.plot_eigenfaces()
variance_ratio = np.cumsum(pca.explained_variance_ratio_)
pca.plot_curve(variance_ratio,
            'Cumulative variance ratio of principal components',
            'Principle component number')

sellect_ratio = np.linspace(variance_ratio.min(),1,7+2)[1:-1]
sellect_feature_index = [np.where(variance_ratio<=ratio)[0][-1] for ratio in sellect_ratio]
sellect_feature_index

test_acc_list = list()
roc_dict = dict()
for i in sellect_feature_index:
    k = i+1
    print('K:',k)
    # load train images
    train_x = dataset['train']['noise'].normal_images.numpy().transpose(0,2,3,1)
    train_x = train_x.reshape(train_x.shape[0],-1)
    train_y = dataset['train']['noise'].labels
    # reconstruct train images
    train_x_recon = pca.getReconstruct(train_x,k)
    pca.plot_image(train_x, 32, 32, 3)
    pca.plot_image(train_x_recon, 32, 32, 3)
    # load test images
    test_x = dataset['test']['noise'].normal_images.numpy().transpose(0,2,3,1)
    test_x = test_x.reshape(test_x.shape[0],-1)
    test_y = dataset['test']['noise'].labels

    _,test_acc,fpr,tpr,auc = classifier(pca,train_x,train_y,test_x,test_y,k)
    test_acc_list.append(test_acc)
    roc_dict[k] = {'fpr':fpr,'tpr':tpr,'auc':auc}

plt.bar(range(1,8),test_acc_list)
plt.show()
plot_roc(roc_dict)
