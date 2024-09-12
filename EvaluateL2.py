#from ModelPlus import ModelPlus
#import DataManagerPytorch_TA as datamanager
from attack_utils import get_model
from attack_utils import get_model, read_imagenet_data_specify, save_results
import argparse
import torch
import numpy as np
import os

model = 'BiT-M'
dataset_used = 'ImageNet'
use_RL = False
loader_created = True
max_queries = None
device = 'cuda'

if use_RL:
    import attack_mask_RL as attack
    max_queries = 500
else:
    import attack_mask as attack
    max_queries = 1000


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", default="results_" + model + ("_RL_" if use_RL else "_") + dataset_used + "_" + str(max_queries), help="Output folder")
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


def validateDA_l2(valLoader, model, l2List, l2Budget, device=None):
    numSamples = len(valLoader.dataset)
    accuracyArray = torch.zeros(numSamples) #variable for keep tracking of the correctly identified samples 
    #switch to evaluate mode
    #model.eval()
    indexer = 0
    accuracy = 0
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume CUDA by default
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device) #use the prefered device if one is specified
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j] or l2List[i] > l2Budget:
                    accuracyArray[indexer] = 1.0 #Mark with a 1.0 if sample is correctly identified
                    accuracy = accuracy + 1
                indexer = indexer + 1 #update the indexer regardless 
    accuracy = accuracy/numSamples
    print("Accuracy:", accuracy)
    return accuracyArray


def Get_Adv_Acc(args, L2_threshold):
    ###############################
    
    print("Load Data")
    val_labels = np.load(os.getcwd() + '/' + args.output_folder + "/Val_Labels.npy", allow_pickle = True)
    adv_labels = np.load(os.getcwd() + '/' + args.output_folder + "/Adv_Labels.npy", allow_pickle = True)
    images, labels, selected_paths = read_imagenet_data_specify(args, device)
    
    # Load the adversarial and validation examples
    adv_examples_file = os.getcwd() + '/' + args.output_folder + "/Adv_Examples.npy"
    val_examples_file = os.getcwd() + '/' + args.output_folder + "/Val_Examples.npy"
    adv_examples = np.load(adv_examples_file)
    val_examples = np.load(val_examples_file)

    val_examples = torch.Tensor.numpy(images.cpu())
    val_labels = torch.Tensor.numpy(labels.cpu())

    # Initialize a list to store the L2 distances
    l2_distances = []

    # Check the L2 distances and store them in the list
    for adv_img, val_img in zip(adv_examples, val_examples):
        l2_distance = np.linalg.norm(adv_img - val_img)
        l2_distances.append(l2_distance)

    # Count how many L2 distances are out of the perturbation limit
    print("L2 Limit:", L2_threshold)

    print("Adv Acc after filtering by l2 budget")
    adv_acc = 0
    for i in range(len(l2_distances)):
        if (l2_distances[i] <= L2_threshold) and (int(val_labels[i]) != int(adv_labels[i].argmax(axis = 0).item())):
            adv_acc += 1 
    print(adv_acc/1000)


if __name__ == '__main__':
    args = get_args()

    # Set side length for determining perturbation limit
    if args.model_name=='inception-v3':
        args.side_length=299
    elif args.model_name == 'ViT-L' or args.model_name == 'BiT-M':
        args.side_length=512
    else:
        args.side_length=224
    print(args.side_length)

    # Iterate through RMSE constants...
    for RMSE in [0.01, 0.05, 0.1]:
        img_size = args.side_length # CIFAR-10 image size, change if different
        L2_threshold = RMSE * np.sqrt(img_size * img_size * 3)
        print("The attack threshold is", L2_threshold)
        Get_Adv_Acc(L2_threshold)
        print("")