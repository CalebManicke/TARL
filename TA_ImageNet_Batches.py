# Import files from Bit and ViT folders
import TransferImageNetV4.BigTransferModels
from TransferImageNetV4.TransformerModels import VisionTransformer, CONFIGS

import json
import torch
import torchvision 
import os
import argparse
import time
from attack_utils import get_model, read_imagenet_data_specify, save_results
from foolbox.distances import l2
import numpy as np
from PIL import Image
import torch_dct

# TA_ImageNet specific libraries
import DataManagerPytorch_TA as datamanager
from ImageNetLoader import LoadImageNetV2
import torchvision.models as models
from foolbox import PyTorchModel
from random import shuffle
import torchvision.transforms.functional as TF

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = 'ViT-L'
use_RL = True
loader_created = True
max_queries = None

if use_RL:
    import attack_mask_RL as attack
    max_queries = 500
else:
    import attack_mask as attack
    max_queries = 1000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results_" + model + ("_RL" if use_RL else "") + "_ImageNet" + "_" + str(max_queries), help="Output folder")
    parser.add_argument(
        "--model_name",
        type=str,
        default=model,
        help="The name of model you want to attack(resnet-18, inception-v3, vgg-16, resnet-101, densenet-121, ViT-B, ViT-L, BiT-M)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="images",
        help="The path of dataset"
    )
    parser.add_argument(
         "--csv",
        type=str,
        default="label.csv",
        help="The path of csv information about dataset"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=20,
        help='The random seed you choose'
    )
    parser.add_argument(
        '--max_queries',
        type=int,
        default=max_queries,
        help='The max number of queries in model'
    )
    parser.add_argument(
        '--ratio_mask',
        type=float,
        default=0.1,
        help='ratio of mask'
    )
    parser.add_argument(
        '--dim_num',
        type=int,
        default=1,
        help='the number of picked dimensions'
    )
    parser.add_argument(
        '--max_iter_num_in_2d',
        type=int,
        default=2,
        help='the maximum iteration number of attack algorithm in 2d subspace'
    )
    parser.add_argument(
        '--init_theta',
        type=int,
        default=2,
        help='the initial angle of a subspace=init_theta*np.pi/32'
    )
    parser.add_argument(
        '--init_alpha',
        type=float,
        default=np.pi/2,
        help='the initial angle of alpha'
    )
    parser.add_argument(
        '--plus_learning_rate',
        type=float,
        default=0.1,
        help='plus learning_rate when success'
    )
    parser.add_argument(
        '--minus_learning_rate',
        type=float,
        default=0.005,
        help='minus learning_rate when fail'
    )
    parser.add_argument(
        '--half_range',
        type=float,
        default=0.1,
        help='half range of alpha from pi/2'
    )
    return parser.parse_args()

# Takes about 10 minutes to convert dimensions for certain models
def get_imagenet(val_dir, side_length, batch_index, batch_size):
    imgs = [file for file in os.listdir(val_dir) if file.endswith(".JPEG")]
    images = torch.zeros((batch_size, 3, side_length, side_length))
    images_index = 0
    for i in range(len(imgs)):
        if batch_index * batch_size <= i < (batch_index + 1) * batch_size:
            # print(os.getcwd() + "//val//val2//ILSVRC2012_val_" + str(i+1).zfill(8) + ".JPEG")
            image = Image.open(os.getcwd() + "//val//val2//ILSVRC2012_val_" + str(i+1).zfill(8) + ".JPEG")
            image = image.convert('RGB')
            image = image.resize((side_length, side_length))
            image = np.asarray(image, dtype=np.float32)
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image) 
            images[images_index] = image / 255
            images_index += 1
    return images


def LoadImageNetLabels(imgNum = 50000):
    # Load the labels
    dir2 = os.getcwd() + "//val.txt"
    file1 = open(dir2)
    Lines = file1.readlines() 
    yData = torch.zeros(50000)
    index = 0
    for line in Lines: 
        splitLine = line.split(" ")
        yData[index] = int(splitLine[1])
        index = index + 1
    return yData


def Main_Attack():
    args = get_args()

    if args.model_name=='inception-v3':
        args.side_length=299
    if args.model_name == 'ViT-L' or args.model_name == 'BiT-M':
        args.side_length=512
    else:
        args.side_length=224

    ###############################
    print("Load Model: %s" % args.model_name)
    fmodel = get_model(args,device)
    
    model_name = args.model_name
    ###############################
    

    ###############################
    batch_index = 0
    batch_size = 64 
    total_num_imgs = 50000
    labels_added = dict()
    for i in range(1000): labels_added[i] = False
    imagenet_labels = LoadImageNetLabels(imgNum = 50000)
    correct_images = torch.zeros((1000, 3, args.side_length, args.side_length))
    correct_labels = torch.zeros((1000))
    correct_index = 0

    
    while (batch_index * batch_size <= total_num_imgs) and correct_index < 1000:
        print("Loading batch #" + str(batch_index))

        # Get labels
        cur_images = None 
        cur_labels = imagenet_labels[batch_index * batch_size: (batch_index + 1) * batch_size]

        # Get current images within current batch
        if model_name == 'inception-v3':
            cur_images = get_imagenet(os.getcwd() + '//val//val2', 299, batch_index, batch_size)
        if model_name == 'ViT-L' or model_name == 'BiT-M':
            cur_images = get_imagenet(os.getcwd() + '//val//val2', 512, batch_index, batch_size)
        else:
            cur_images = get_imagenet(os.getcwd() + '//val//val2', 224, batch_index, batch_size)

        # Get correctly identified examples
        cur_imagenet_loader = datamanager.TensorToDataLoader(xData = cur_images, yData = cur_labels, batchSize = 64)
        acc_array, acc = datamanager.validateDA(cur_imagenet_loader, fmodel, device, False, False)

        # Add correctly identified examples to 1000 long tensor if we don't have another example of that exact class
        for i in range(len(acc_array)):
            # Increase current index
            cur_label_index = batch_index * batch_size + i

            # Add correctly identified example if it comes from a new class
            if int(acc_array[i]) == 1 and (not labels_added[int(cur_labels[i])]):
                correct_images[correct_index] = cur_images[i]
                correct_labels[correct_index] = cur_labels[i]
                labels_added[int(cur_labels[i])] = True
                correct_index += 1
                print(str(int(cur_labels[i])) + " Added!")

        # Free up CPU memory
        batch_index += 1
        del cur_imagenet_loader 
        del cur_images 
        del cur_labels
    
    
    # Note: Whatever CPU Caiwen has in CaiwenML should be able to handle a 1000 image long tensor...
    imagenet_loader = datamanager.TensorToDataLoader(xData = correct_images, yData = correct_labels, batchSize = 1)

    # Transform images for model mean and std... no transforms library in Pytorch 1.7.1 means we have to normalize each image manually!!!
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    
    ###############################

    # Note: Everything is deterministic up to this point, we should be able to run this again to get wrong class labels...
    #val_loader, wrong_classes = datamanager.GetCorrectlyIdentifiedSamplesBalanced(model = fmodel, totalSamplesRequired = 1000, dataLoader = imagenet_loader, numClasses = 1000)
    images, labels = datamanager.DataLoaderToTensor(imagenet_loader)
    
    ###############################
    print("Attack !")
    time_start = time.time()

    my_intermediates = []
    my_labels_adv = []
    my_adv_l2 = []
    my_advs = torch.zeros((1000, 3, args.side_length, args.side_length))

    ta_model = attack.TA(fmodel, input_device=device)
    for i in range(len(images)):
        cur_image = images[i].unsqueeze(0).to(device)
        cur_label = labels[i].unsqueeze(0).to(device)
        cur_adv, q_list, cur_intermediates, max_length = ta_model.attack(args, cur_image, cur_label)
        print('TA Attack Done')
        print("{:.2f} s to run".format(time.time() - time_start))
        print("Results")

        cur_label_adv = fmodel(cur_adv).argmax(1)
        cur_adv_l2 = l2(cur_image, cur_adv)
        my_advs[i] = cur_adv[0]
        for cur_intermediate in cur_intermediates:
            my_intermediates.append(cur_intermediate)
        my_labels_adv.append(cur_label_adv[0].cpu())
        my_adv_l2.append(cur_adv_l2[0].cpu())
    '''
    for image_i in range(1, len(images)):
        print("My Adversarial Image {}:".format(image_i))
        label_o = int(labels[image_i])
        label_adv = int(my_labels_advs[image_i])
        print("\t- l2 = {}".format(my_advs_l2[image_i]))
        print("\t- {} queries\n".format(q_list[image_i]))
    '''
    print(my_intermediates)
    save_results(args,my_intermediates, len(images))

    np.save(os.getcwd() + "/" + args.output_folder + "/Val_Examples.npy", images.cpu().numpy())
    np.save(os.getcwd() + "/" + args.output_folder + "/Val_Labels.npy", labels.cpu().numpy())
    np.save(os.getcwd() + "/" + args.output_folder + "/Adv_l2_Distances.npy", np.array(my_adv_l2))
    np.save(os.getcwd() + "/" + args.output_folder + "/Adv_Labels.npy", np.array(my_labels_adv))
    np.save(os.getcwd() + "/" + args.output_folder + "/Adv_Examples.npy", my_advs.cpu().numpy())
    #np.save(os.getcwd() + "/" + args.output_folder + "/Wrong_Class_Labels.npy", np.array(wrong_classes)) 

    ###############################
    print("Test Model")
    _, valAcc = datamanager.validateDA(imagenet_loader, fmodel, device=device)
    adv_loader = datamanager.TensorToDataLoader(xData = my_advs, yData = labels, batchSize = 1)
    _, advAcc = datamanager.validateDA(adv_loader, fmodel, device=device)
    print("Val Acc: " + str(valAcc))
    print("Adv Acc: " + str(advAcc))


def Get_Adv_Acc():
    args = get_args()
    if args.model_name=='inception-v3':
        args.side_length=299
    if args.model_name == 'ViT-L' or args.model_name == 'BiT-M':
        args.side_length=512
    else:
        args.side_length=224

    print(args.max_queries)

    ###############################
    print("Load Model: %s" % args.model_name)
    fmodel = get_model(args,device)

    ###############################
    

    ###############################
    print("Load Data")
    val_labels = np.load(os.getcwd() + "/" + args.output_folder + "/Val_Labels.npy")
    adv_examples = np.load(os.getcwd() + "/" + args.output_folder + "/Adv_Examples.npy")
    cur_index = 0
    adv_acc = 0
    
    while cur_index < 1000:
        print("Loading Image #" + str(cur_index))

        # Get label
        cur_label = int(val_labels[cur_index])
        cur_image = torch.from_numpy(adv_examples[cur_index])
        cur_image = cur_image.unsqueeze(0).to(device)

        # Check model output 
        output_label = fmodel(cur_image)[0]
        output_label = output_label.argmax(axis = 0)
        if int(output_label) != cur_label:
            adv_acc += 1
        
        cur_index += 1

    print("Total Adversarial Accuracy: " + str((float(1000 - adv_acc) / 1000) * 100) + "% Robustness")


def Get_Val_Acc():
    args = get_args()
    if args.model_name=='inception-v3':
        args.side_length=299
    if args.model_name == 'ViT-L' or args.model_name == 'BiT-M':
        args.side_length=512
    else:
        args.side_length=224
    print(args.max_queries)

    ###############################
    print("Load Model: %s" % args.model_name)
    fmodel = get_model(args,device)

    ###############################
    

    ###############################
    print("Load Data")
    cur_index = 0
    imagenet_labels = LoadImageNetLabels(imgNum = 50000)
    total_acc = 0
    
    while cur_index < 50000:
        print("Loading Image #" + str(cur_index))

        # Get label
        cur_label = imagenet_labels[cur_index]
        cur_image = None

        # Get current images within current batch
        if args.model_name == 'inception-v3':
            cur_image = get_imagenet(os.getcwd() + '//val//val2', 299, cur_index, 1)
        if args.model_name == 'ViT-L' or args.model_name == 'BiT-M':
            cur_image = get_imagenet(os.getcwd() + '//val//val2', 512, cur_index, 1)
        else:
            cur_image = get_imagenet(os.getcwd() + '//val//val2', 224, cur_index, 1)
        
        cur_image = cur_image.to(device)
        output = fmodel(cur_image)
        if int(output[0].argmax(axis=0)) == int(cur_label):
            total_acc += 1
        
        cur_index += 1

    print("Total Validation Accuracy: " + str(float(total_acc) / 50000))

if __name__ == '__main__':
    #Main_Attack()
    Get_Adv_Acc()
    #Get_Val_Acc()