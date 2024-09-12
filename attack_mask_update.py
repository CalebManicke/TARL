# This is the attack mask with a different alpha update

from foolbox.attacks.blended_noise import LinearSearchBlendedUniformNoiseAttack
from foolbox.attacks.base import MinimizationAttack, get_criterion
import sys
import torch_dct
from attack_utils import *
import time
import numpy as np
from bayes_opt import BayesianOptimization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from bayes_opt.util import UtilityFunction, acq_max, ensure_rng

global device


# initialize an adversarial example with uniform noise
def get_x_adv(x_o: torch.Tensor, label: torch.Tensor, model) -> torch.Tensor:
    criterion = get_criterion(label)
    init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=100)
    x_adv = init_attack.run(model, x_o, criterion)
    return x_adv


# coompute the difference
def get_difference(x_o: torch.Tensor, x_adv: torch.Tensor) -> torch.Tensor:
    difference = x_adv - x_o
    if torch.norm(difference, p=2) == 0:
        raise ('difference is zero vector!')
        return difference
    return difference


def rotate_in_2d(x_o2x_adv: torch.Tensor, direction: torch.Tensor, theta: float = np.pi / 8) -> torch.Tensor:
    alpha = torch.sum(x_o2x_adv * direction) / torch.sum(x_o2x_adv * x_o2x_adv)
    orthogonal = direction - alpha * x_o2x_adv
    direction_theta = x_o2x_adv * np.cos(theta) + torch.norm(x_o2x_adv, p=2) / torch.norm(orthogonal,
                                                                                          p=2) * orthogonal * np.sin(
        theta)
    direction_theta = direction_theta / torch.norm(direction_theta) * torch.norm(x_o2x_adv)
    return direction_theta


# obtain the mask in the low frequency
def get_orthogonal_1d_in_subspace(args,x_o2x_adv: torch.Tensor, n, ratio_size_mask=0.3, if_left=1) -> torch.Tensor:
    random.seed(time.time())
    zero_mask = torch.zeros(size=[args.side_length, args.side_length], device=device)
    size_mask = int(args.side_length * ratio_size_mask)
    if if_left:
        zero_mask[:size_mask, :size_mask] = 1

    else:
        zero_mask[-size_mask:, -size_mask:] = 1

    to_choose = torch.where(zero_mask == 1)
    x = to_choose[0]
    y = to_choose[1]

    select = np.random.choice(len(x), size=n, replace=False)
    mask1 = torch.zeros_like(zero_mask)
    mask1[x[select], y[select]] = 1
    mask1 = mask1.reshape(-1, args.side_length, args.side_length)

    select = np.random.choice(len(x), size=n, replace=False)
    mask2 = torch.zeros_like(zero_mask)
    mask2[x[select], y[select]] = 1
    mask2 = mask2.reshape(-1, args.side_length, args.side_length)

    select = np.random.choice(len(x), size=n, replace=False)
    mask3 = torch.zeros_like(zero_mask)
    mask3[x[select], y[select]] = 1
    mask3 = mask3.reshape(-1, args.side_length, args.side_length)

    mask = torch.cat([mask1, mask2, mask3], dim=0).expand([1, 3, args.side_length, args.side_length])
    mask *= torch.randn_like(mask, device=device)
    direction = rotate_in_2d(x_o2x_adv, mask, theta=np.pi / 2)
    return direction / torch.norm(direction, p=2) * torch.norm(x_o2x_adv, p=2), mask


# compute the best adversarial example in the surface
def get_x_hat_in_2d(x_o: torch.Tensor, x_adv: torch.Tensor, axis_unit1: torch.Tensor, axis_unit2: torch.Tensor,
                    net: torch.nn.Module, queries, original_label, max_iter=2,plus_learning_rate=0.01,minus_learning_rate=0.0005,half_range=0.5, init_alpha = np.pi/2, history = []):
    
    # Took an additional parameter history[]

    # We already secured the 2-D space
    if not hasattr(get_x_hat_in_2d, 'alpha'):
        get_x_hat_in_2d.alpha = init_alpha # Getting intial value of alpha
    
    # Upper and Lower bound of Alpha - Could be useful for our RL model
    upper = np.pi / 2 + half_range 
    lower = np.pi / 2 - half_range

    # Getting the l2 value between x_adv and x_o
    d = torch.norm(x_adv - x_o, p=2) 

    # Initialize beta
    theta = max(np.pi - 2 * get_x_hat_in_2d.alpha, 0) + min(np.pi / 16, get_x_hat_in_2d.alpha / 2) 

    # Convert x_adv into frequency domain representation using DCT
    x_hat = torch_dct.idct_2d(x_adv)

    # Set right_theta to be (pi - alpha)
    right_theta = np.pi - get_x_hat_in_2d.alpha 

    # They are finding the adversarial example with alpha and beta
    x = x_o + d * (axis_unit1 * np.cos(theta) + axis_unit2 * np.sin(theta)) / np.sin(get_x_hat_in_2d.alpha) * np.sin(
        get_x_hat_in_2d.alpha + theta)

    # Scaled DCT-III done through the last dimension, then some pixel work
    x = torch_dct.idct_2d(x) 
    get_x_hat_in_2d.total += 1 
    get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
    x = torch.clamp(x, 0, 1) 

    # Get the label for x
    label = get_label(net(x))
    dist = torch.norm(x_o - x) #Might not work becuase all the pixel work we did on x
    queries += 1
    
    # If x adversarial with beta value
    if label != original_label: 
        x_hat = x
        left_theta = theta
        flag = 1
        history.append((get_x_hat_in_2d.alpha, dist.item(),  True))
    
    # If x not adversarial with beta value
    else:
        # Store the failure
        history.append((get_x_hat_in_2d.alpha, dist.item(),  False))

        # Update alpha with equation 3
        get_x_hat_in_2d.alpha -= minus_learning_rate # minus_learning_rate=0.0005
        get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)

        # Update beta as well
        theta = max(theta, np.pi - 2 * get_x_hat_in_2d.alpha + np.pi / 64)

        # Generate a new adversarial example after updating and do some pixel work
        x = x_o + d * (axis_unit1 * np.cos(theta) - axis_unit2 * np.sin(theta)) / np.sin(
            get_x_hat_in_2d.alpha) * np.sin(
            get_x_hat_in_2d.alpha + theta) 
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)

        # Get the label this time
        label = get_label(net(x))
        dist = torch.norm(x_o - x)
        queries += 1
        

        # If the second one is adversarial
        if label != original_label:
            x_hat = x
            left_theta = theta
            flag = -1
            # Store the success
            history.append((get_x_hat_in_2d.alpha, dist.item(),  True))

        # If the both are not adversarial
        else:
            # Store the failure and Update Alpha
            history.append((get_x_hat_in_2d.alpha, dist.item(),  False)) 
            get_x_hat_in_2d.alpha -= minus_learning_rate 
            get_x_hat_in_2d.alpha = max(get_x_hat_in_2d.alpha, lower)
            # Return the original adversarial input, queries used,
            # and the label False indicating we did not change the adversarial example
            return x_hat, queries, False

    # Proceed to binary search for beta as long as one of the examples is adversarial
    theta = (left_theta + right_theta) / 2

    # Objective function for Basysian Optimization, want to maximize -score
    def objective(alpha):
        filtered_data = [t for t in history if t[2]]
        l2_distances = [t[1] for t in filtered_data]
        angle_differences = [abs(alpha - t[0]) for t in filtered_data]
        score = sum([1 / l2 * angle_diff for l2, angle_diff in zip(l2_distances, angle_differences)])
        return -score

    # Setting up alpha update model and ranges, must have potential to be hypertuned
    bounds = {'alpha': (lower, upper)}
    bo = BayesianOptimization(f=objective, pbounds=bounds, random_state=42)
    bo.maximize(init_points = 5, n_iter=0) # Modify this init_point later
    util = UtilityFunction(kind='ucb', kappa=2.5, xi=0.0)

    model = Sequential([Dense(32, activation='relu', input_shape=(1,)),
                        Dropout(0.5),
                        Dense(1, activation='linear')
                        ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    def train_and_predict(X, y):
        model.fit(X, y, epochs=10, verbose=0)
        return -model.predict(np.array([[X[-1]]]))[0][0]

    # Search for the best adversarial example in the subspace
    for i in range(max_iter): #max_iter is set to 2?

        # New candidate x, previous candidate is x_hat
        x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
            get_x_hat_in_2d.alpha) * np.sin(
            get_x_hat_in_2d.alpha + theta)
        
        # Scaling and pixel work
        x = torch_dct.idct_2d(x)
        get_x_hat_in_2d.total += 1
        get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
        x = torch.clamp(x, 0, 1)

        # Get the label for next candidate
        label = get_label(net(x))
        dist = torch.norm(x_o - x)
        queries += 1

        # If x is adversarial
        if label != original_label:
            # Store the success
            history.append((get_x_hat_in_2d.alpha, dist.item(),  True))
            left_theta = theta
            x_hat = x

            # Update alpha
            next_params = bo.suggest(util)
            next_alpha = next_params['alpha']
            score = objective(next_alpha)
            bo.register(params=next_params, target=score)
            X = np.array([obs['params']['alpha'] for obs in bo.res])
            y = np.array([obs['target'] for obs in bo.res])
            train_and_predict(X, y)
            get_x_hat_in_2d.alpha = bo.max['params']['alpha']
            best_score = bo.max['target']

            # Old alpha update
            #get_x_hat_in_2d.alpha += plus_learning_rate 
            return x_hat, queries, True

        # If x is not adversarial
        else:

            next_params = bo.suggest(util)
            next_alpha = next_params['alpha']
            score = objective(next_alpha)
            bo.register(params=next_params, target=score)
            X = np.array([obs['params']['alpha'] for obs in bo.res])
            y = np.array([obs['target'] for obs in bo.res])
            train_and_predict(X, y)
            get_x_hat_in_2d.alpha = bo.max['params']['alpha']
            best_score = bo.max['target']

            # Old alpha update according to equation 3, change this to our method
            #get_x_hat_in_2d.alpha -= minus_learning_rate 
            #get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha)
            theta = max(theta, np.pi - 2 * get_x_hat_in_2d.alpha + np.pi / 64)

            flag = -flag

            # New candidate
            x = x_o + d * (axis_unit1 * np.cos(theta) + flag * axis_unit2 * np.sin(theta)) / np.sin(
                get_x_hat_in_2d.alpha) * np.sin(
                get_x_hat_in_2d.alpha + theta)

            # Scaling and pixel work
            x = torch_dct.idct_2d(x)
            get_x_hat_in_2d.total += 1
            get_x_hat_in_2d.clamp += torch.sum(x > 1) + torch.sum(x < 0)
            x = torch.clamp(x, 0, 1)

            # Get the label for this new x
            label = get_label(net(x))
            dist = torch.norm(x_o - x)
            queries += 1

            # If new candidate is adversarial
            if label != original_label:
                left_theta = theta
                x_hat = x

                #Save this success
                history.append((get_x_hat_in_2d.alpha, dist.item(),  True))

                # Update alpha
                next_params = bo.suggest(util)
                next_alpha = next_params['alpha']
                score = objective(next_alpha)
                bo.register(params=next_params, target=score)
                X = np.array([obs['params']['alpha'] for obs in bo.res])
                y = np.array([obs['target'] for obs in bo.res])
                train_and_predict(X, y)
                get_x_hat_in_2d.alpha = bo.max['params']['alpha']
                best_score = bo.max['target']
                # Update alpha, change this 
                #get_x_hat_in_2d.alpha += plus_learning_rate # change this here into our method
                #get_x_hat_in_2d.alpha = min(upper, get_x_hat_in_2d.alpha) #change this into our method
                return x_hat, queries, True
            
            # None of the two steps we took is adversarial 
            else:

                # Update alpha
                next_params = bo.suggest(util)
                next_alpha = next_params['alpha']
                score = objective(next_alpha)
                bo.register(params=next_params, target=score)
                X = np.array([obs['params']['alpha'] for obs in bo.res])
                y = np.array([obs['target'] for obs in bo.res])
                train_and_predict(X, y)
                get_x_hat_in_2d.alpha = bo.max['params']['alpha']
                best_score = bo.max['target']
                # get_x_hat_in_2d.alpha -= minus_learning_rate # change this
                # get_x_hat_in_2d.alpha = max(lower, get_x_hat_in_2d.alpha) # change this
                left_theta = max(np.pi - 2 * get_x_hat_in_2d.alpha, 0) + min(np.pi / 16, get_x_hat_in_2d.alpha / 2)
                right_theta = theta
            
        theta = (left_theta + right_theta) / 2

    # After looping max_iter times

    # Update alpha again
    next_params = bo.suggest(util)
    next_alpha = next_params['alpha']
    score = objective(next_alpha)
    bo.register(params=next_params, target=score)
    X = np.array([obs['params']['alpha'] for obs in bo.res])
    y = np.array([obs['target'] for obs in bo.res])
    train_and_predict(X, y)
    get_x_hat_in_2d.alpha = bo.max['params']['alpha']
    best_score = bo.max['target']
    # get_x_hat_in_2d.alpha += plus_learning_rate
    # get_x_hat_in_2d.alpha = min(upper, get_x_hat_in_2d.alpha)
    return x_hat, queries, True



# Main TA method from pseducode
def get_x_hat_arbitary(args,x_o: torch.Tensor, net: torch.nn.Module, original_label, init_x=None,dim_num=5):
    
    # If the input org image is adversarial
    if get_label(net(x_o)) != original_label: 
        return x_o, 1001, [[0, 0.], [1001, 0.]]
    
    # Initialize a large adversarial example
    if init_x is None:
        x_adv = get_x_adv(x_o, original_label, net) 
    else:
        x_adv = init_x
    

    x_hat = x_adv
    queries = 0.
    dist = torch.norm(x_o - x_adv)
    intermediate = []
    intermediate.append([0, dist.item(), get_x_hat_in_2d.alpha])

    # Historical Data for reinforcement model training
    history = []
    # Append the alpha value, l2 value, and the adversarial label
    history.append((get_x_hat_in_2d.alpha, dist.item(),  True)) 

    while queries < args.max_queries :

        # Sampling 2-D subspace in the low frequency space
        x_o2x_adv = torch_dct.dct_2d(get_difference(x_o, x_adv))
        axis_unit1 = x_o2x_adv / torch.norm(x_o2x_adv) # Unit Vector

        # Obtain the mask in the low frequency
        direction, mask = get_orthogonal_1d_in_subspace(args,x_o2x_adv, dim_num, args.ratio_mask, args.dim_num)
        axis_unit2 = direction / torch.norm(direction)


        x_hat, queries, changed = get_x_hat_in_2d(torch_dct.dct_2d(x_o), torch_dct.dct_2d(x_adv), axis_unit1,
                                                  axis_unit2, net, queries, original_label, max_iter=args.max_iter_num_in_2d,
                                                  plus_learning_rate=args.plus_learning_rate,minus_learning_rate=args.minus_learning_rate,half_range=args.half_range, init_alpha=args.init_alpha,
                                                  history = history)
        x_adv = x_hat

        dist = torch.norm(x_hat - x_o)
        intermediate.append([queries, dist.item(), get_x_hat_in_2d.alpha])
        if queries >= args.max_queries:
            break
    return x_hat, queries, intermediate




class TA:
    def __init__(self, model, input_device):
        self.net = model
        global device
        device = input_device

    def attack(self, args,inputs, labels):
        x_adv_list = torch.zeros_like(inputs)
        queries = []
        intermediates = []
        l2 = []
        normalized_l2 = []
        init_attack: MinimizationAttack = LinearSearchBlendedUniformNoiseAttack(steps=50)
        criterion = get_criterion(labels.long())
        best_advs = init_attack.run(self.net, inputs, criterion, early_stop=None)
        max_length = 0
        acc = [0., 0., 0.]
        for i, [input, label] in enumerate(zip(inputs, labels)):
            print('[{}/{}]:'.format(i + 1, len(inputs)), end='')
            global probability
            probability = np.ones(input.shape[1] * input.shape[2])
            global is_visited_1d
            is_visited_1d = torch.zeros(input.shape[0] * input.shape[1] * input.shape[2])
            global selected_h
            global selected_w
            selected_h = input.shape[1]
            selected_w = input.shape[2]
            get_x_hat_in_2d.alpha = np.pi / 2

            get_x_hat_in_2d.total = 0
            get_x_hat_in_2d.clamp = 0

            x_adv, q, intermediate = get_x_hat_arbitary(args,input[np.newaxis, :, :, :], self.net,
                                                        label.reshape(1, ).to(device),
                                                        init_x=best_advs[i][np.newaxis, :, :, :], dim_num=args.dim_num)
            x_adv_list[i] = x_adv[0]
            diff = torch.norm(x_adv[0] - input, p=2) / (args.side_length * np.sqrt(3))
            if diff <= 0.1:
                acc[0] += 1
            if diff <= 0.05:
                acc[1] += 1
            if diff <= 0.01:
                acc[2] += 1
            print("Top-1 Acc:{} Top-2 Acc:{} Top-3 Acc:{}".format(acc[0] / (i + 1), acc[1] / (i + 1),
                                                                          acc[2] / (i + 1)))
            l2.append(torch.norm(x_adv[0] - input, p=2))
            normalized_l2.append(diff)                                                       
            print("List of l2: ", l2)
            print("List of Normalized L2: ", normalized_l2)
            queries.append(q)
            intermediates.append(intermediate)
            if max_length < len(intermediate):
                max_length = len(intermediate)
        queries = np.array(queries)
        return x_adv_list, queries, intermediates, max_length