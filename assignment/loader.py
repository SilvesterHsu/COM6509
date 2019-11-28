import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(1720410)
IMAGE_SIZE = torch.Size([3,32,32])
NOISE = torch.rand(IMAGE_SIZE)
SCALE = 0.3
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
        assert len(self.images) == len(self.labels)

    def __loaddata(self, target_classes):
        dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=self.train, download=self.download, transform=None)
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        for image, label in dataset:
            if classes[label] in target_classes:
                self.images.append(image)
                self.labels.append(target_classes.index(classes[label]))

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
trainset = {'raw':CIFAR10Dataset(classes, root='./data', train=True,
                     download=True, transform=transform['raw']),
            'noise':CIFAR10Dataset(classes, root='./data', train=True,
                                 download=True, transform=transform['noise'])
            }

trainloader = {'raw':torch.utils.data.DataLoader(trainset['raw'], batch_size=BATCH,
                                          shuffle=False, num_workers=2),
               'noise':torch.utils.data.DataLoader(trainset['noise'], batch_size=BATCH,
                                                         shuffle=False, num_workers=2)
            }


images, labels = next(iter(trainloader['raw']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))

images, labels = next(iter(trainloader['noise']))
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(BATCH)))


for p in ['raw','noise']:
    c,cat_tensor = 0,torch.zeros([10,3,32,32])
    d,dog_tensor = 0,torch.zeros([10,3,32,32])
    for image,label in trainset[p]:
        if classes[label] == 'cat' and c<10:
            cat_tensor[c] = image
            c += 1
        if classes[label] == 'dog' and d<10:
            dog_tensor[d] = image
            d += 1
    print(p)
    imshow(torchvision.utils.make_grid(cat_tensor))
    imshow(torchvision.utils.make_grid(dog_tensor))
