# Import files from Bit and ViT folders
import sys
sys.path.insert(0, '/data4/cam18027/TriangleAttack/TransferImageNetV4/TransferImageNetV4')
import BigTransferModels
from TransformerModels import VisionTransformer, CONFIGS

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
model = 'BiT-M'
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
    parser.add_argument("--output_folder", "-o", default="results_" + model + ("_RL" if use_RL else "") +  "_" + str(max_queries), help="Output folder")
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
    cur_index = 0
    correct_count = 0
    images, labels, selected_paths = read_imagenet_data_specify(args, device)
    print("{} images loaded with the following labels: {}".format(len(images), labels))

    print("Attack !")
    time_start = time.time()

    my_intermediates = []
    my_labels_adv = []
    my_adv_l2 = []
    my_advs = torch.zeros((200, 3, args.side_length, args.side_length))

    ta_model = attack.TA(fmodel, input_device=device)

    
    while cur_index < len(images):
        # Get correctly identified examples
        output = fmodel(images[cur_index].unsqueeze(0)) 
        output_label = output[0].argmax(axis = 0)
        print(output_label)
        print(labels[cur_index].item())
        if int(output_label) == int(labels[cur_index].item()): correct_count += 1

        cur_image = images[cur_index].unsqueeze(0).to(device)
        cur_label = labels[cur_index].unsqueeze(0).to(device)
        cur_adv, q_list, cur_intermediates, max_length = ta_model.attack(args, cur_image, cur_label)
        print('TA Attack Done')
        print("{:.2f} s to run".format(time.time() - time_start))
        print("Results")

        cur_label_adv = fmodel(cur_adv).argmax(1)
        cur_adv_l2 = l2(cur_image, cur_adv)
        my_advs[cur_index] = cur_adv[0]
        for cur_intermediate in cur_intermediates:
            my_intermediates.append(cur_intermediate)
        my_labels_adv.append(cur_label_adv[0].cpu())
        my_adv_l2.append(cur_adv_l2[0].cpu())

        cur_index += 1

    print(my_intermediates)
    save_results(args,my_intermediates, len(images))

    np.save(os.getcwd() + "/" + args.output_folder + "/Val_Examples.npy", np.array(images))
    np.save(os.getcwd() + "/" + args.output_folder + "/Adv_l2_Distances.npy", np.array(my_adv_l2))
    np.save(os.getcwd() + "/" + args.output_folder + "/Adv_Labels.npy", np.array(my_labels_adv))
    np.save(os.getcwd() + "/" + args.output_folder + "/Adv_Examples.npy", my_advs.cpu().numpy())
    np.save(os.getcwd() + "/" + args.output_folder + "/Val_Labels.npy", labels.cpu().numpy())
    #np.save(os.getcwd() + "/" + args.output_folder + "/Wrong_Class_Labels.npy", np.array(wrong_classes)) 

    ###############################
    print("Test Model")
    print("Val Acc: " + str(correct_count / 200))


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
    
    while cur_index < 200:
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

    print("Total Adversarial Accuracy: " + str((float(200 - adv_acc) / 200) * 100) + "% Robustness")


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
    images, labels, selected_paths = read_imagenet_data_specify(args, device)
    total_acc = 0
    
    while cur_index < 200:
        print("Loading Image #" + str(cur_index))

        # Get label
        cur_label = int(labels[cur_index])
        cur_image = images[cur_index].unsqueeze(0).to(device)

        # Get model output
        output = fmodel(cur_image)
        if int(output[0].argmax(axis=0)) == int(cur_label):
            total_acc += 1
        
        cur_index += 1

    print("Total Validation Accuracy: " + str(float(total_acc) / 200))
    

if __name__ == '__main__':
    Main_Attack()
    Get_Adv_Acc()
    #Get_Val_Acc()