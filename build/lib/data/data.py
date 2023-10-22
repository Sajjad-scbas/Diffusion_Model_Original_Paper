import torch 
import torchvision
import matplotlib.pyplot as plt


def get_data(name, transform, root = '.'):
    if name == 'MNIST':
        return torchvision.datasets.MNIST(root=root, download=True, transform=transform)
    elif name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(root=root, download=True, transform=transform)
    elif name == 'StanfordCars':
        return torchvision.datasets.StanfordCars(root=root, download=True, transform=transform)
    elif name == 'Flowers102':
        return torchvision.datasets.Flowers102(root=root, download=True, transform=transform)
    else : 
        raise ValueError('Unsupported dataset name')

def show_images(dataset, num_samples=10, cols=4):
    plt.figure(figsize=(15,15))
    for idx, img in zip(range(num_samples), dataset):
        plt.subplot(int(num_samples//cols)+1, cols, idx+1)
        plt.imshow(img[0])

if __name__ == "__main__":
    root = './data/'
    data = get_data('Flowers102', root)
    show_images(data)
    print('Done!')
    