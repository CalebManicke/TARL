import os
import torch
import torchvision
import DataManagerPytorch as DMP

#This method resizes all the images to 224x224 before processing
#It SHOULD NOT be called to measure clean accuracy of networks that need bigger (e.g. 384) sizes
#This is because going orig res => 224x224 => 384x384 will result in loss of info as opposed to
#Going orig res => 384x384
def LoadImageNetV2(imgNum = 50000, imgSize = 224, batchSize = 16):
    #Load the images 
    dir = os.getcwd() + 'ImageNet//val//'
    dir2 = os.getcwd() + "//ImageNet//labels//val.txt"
    #First load the data 
    xData = torch.zeros(imgNum, 3, imgSize, imgSize)
    t = torchvision.transforms.ToTensor()
    p = torchvision.transforms.ToPILImage()
    rs = torchvision.transforms.Resize((imgSize, imgSize))
    valData = torchvision.datasets.ImageFolder(root=dir, transform=None)
    for i in range(0, imgNum):
        x = rs(valData[i][0])
        xData[i] = t(x)
        if i % 1000 == 0:
            print("Loaded up to image:", i)
    #Next load the labels 
    file1 = open(dir2)
    Lines = file1.readlines() 
    yData = torch.zeros(50000)
    index = 0
    for line in Lines: 
        splitLine = line.split(" ")
        yData[index] = int(splitLine[1])
        index = index + 1
    finalLoader = DMP.TensorToDataLoader(xData, yData[0:imgNum], transforms = None, batchSize = batchSize, randomizer = None)
    print("ImageNet Load Complete.")
    return finalLoader

#This gets the image data without directly loading it and the labels 
def GetRawImageNet():
    dir = os.getcwd() + 'ImageNet//val//'
    dir2 = os.getcwd() + "//ImageNet//labels//val.txt"
    #First load the data 
    valData = torchvision.datasets.ImageFolder(root=dir, transform=None)
    #Next load the labels 
    file1 = open(dir2)
    Lines = file1.readlines() 
    yData = torch.zeros(50000)
    index = 0
    for line in Lines: 
        splitLine = line.split(" ")
        yData[index] = int(splitLine[1])
        index = index + 1
    return valData, yData

def LoadImageNetWithCrop(imgNum = 50000, imgSize = 512, cropSize= 384, batchSize = 16):
    #Load the images 
    dir = os.getcwd() + 'ImageNet//val//'
    dir2 = os.getcwd() + "//ImageNet//labels//val.txt"
    #First load the data 
    xData = torch.zeros(imgNum, 3, cropSize, cropSize)
    t = torchvision.transforms.ToTensor()
    p = torchvision.transforms.ToPILImage()
    rs = torchvision.transforms.Resize((imgSize, imgSize))
    c = torchvision.transforms.CenterCrop((cropSize, cropSize))
    valData = torchvision.datasets.ImageFolder(root=dir, transform=None)
    for i in range(0, imgNum):
        x = rs(valData[i][0]) #resize
        x = c(x) #crop
        xData[i] = t(x) #convert to tensor and save
        if i % 1000 == 0:
            print("Loaded up to image:", i)
    #Next load the labels 
    file1 = open(dir2)
    Lines = file1.readlines() 
    yData = torch.zeros(50000)
    index = 0
    for line in Lines: 
        splitLine = line.split(" ")
        yData[index] = int(splitLine[1])
        index = index + 1
    finalLoader = DMP.TensorToDataLoader(xData, yData[0:imgNum], transforms = None, batchSize = batchSize, randomizer = None)
    print("ImageNet Load Complete.")
    return finalLoader