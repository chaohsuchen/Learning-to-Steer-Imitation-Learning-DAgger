import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import os

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

# Added
from dataset_loader import BalancedBatchSampler

def train_discrete(model, iterator, opt, args):
    model.train()

    loss_hist = []

    # Do one pass over the data accessed by the training iterator
    # Upload the data in each batch to the GPU (if applicable)
    # Zero the accumulated gradient in the optimizer
    # Compute the cross_entropy loss with and without weights
    # Compute the derivatives of the loss w.r.t. network parameters
    # Take a step in the approximate gradient direction using the optimizer opt

    for i_batch, batch in enumerate(iterator):

        x = batch['image']
        y = batch['cmd']

        # CPU or sent to GPU
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # zero accumulated gradient for each round
        opt.zero_grad()

        logits = model(x)
        # args.class_dist = [0.7, 0.2, 0.1] -> class 0 appears 70% of the time
        # after inversion: tensor([1.42, 5.0, 10.0]) -> making mistake on class 2 matters more!
        if args.weighted_loss:
            class_weights = torch.tensor(1.0 / (args.class_dist + 1e-8), dtype=torch.float32).to(DEVICE) # add 1e-8 to prevent dividing 0
            loss = F.cross_entropy(logits, y, weight=class_weights)
            # logits = [1.0, 0.5, 2.0] target = 2
            # p(class 0) = 0.231 p(class 1) = 0.140 p(class 2) = 0.629 but the correct class is 2
            # -log(0.629) ~= 0.464 but class 2 has weight of 10 so 0.464 * 10 = 4.64
        else:
            loss = F.cross_entropy(logits, y)

        loss.backward()
        opt.step()

        loss = loss.detach().cpu().numpy()
        loss_hist.append(loss)

        PRINT_INTERVAL = int(len(iterator) / 3)
        if (i_batch + 1) % PRINT_INTERVAL == 0:
            print ('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t Time: {:10.3f}'.format(
                i_batch, len(iterator),
                i_batch / len(iterator) * 100,
                np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0),
                time.time() - args.start_time,
            ))


def accuracy(y_pred, y_true):
    "y_true is (batch_size) and y_pred is (batch_size, K)"
    _, y_max_pred = y_pred.max(1)
    correct = ((y_true == y_max_pred).float()).mean()
    acc = correct * 100
    return acc


def test_discrete(model, iterator, opt, args):
    model.train()

    acc_hist = []

    for i_batch, batch in enumerate(iterator):
        x = batch['image']
        y = batch['cmd']

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        y_pred = F.softmax(logits, 1)

        acc = accuracy(y_pred, y)
        acc = acc.detach().cpu().numpy()
        acc_hist.append(acc)

    avg_acc = np.asarray(acc_hist).mean()

    print ('\tVal: \tAcc: {}  Time: {:10.3f}'.format(
        avg_acc,
        time.time() - args.start_time,
    ))

    return avg_acc

def get_class_distribution(iterator, args):
    class_dist = np.zeros((args.n_steering_classes,), dtype=np.float32)
    for i_batch, batch in enumerate(iterator):
        y = batch['cmd'].detach().numpy().astype(np.int32) # cmd: class labels
        class_dist += np.bincount(y, minlength=args.n_steering_classes)  # counts # times each class appears

    return (class_dist / sum(class_dist))


def main(args):

    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])

    training_dataset = DrivingDataset(root_dir=args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)

    # I think validation should not transform images color and do randomrotation. This is unfair.
    # So I added this:
    val_transform = transforms.Compose([ transforms.ToPILImage(),
                                         transforms.ToTensor()])

    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=val_transform)


    # training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    # I want to speed up, so I select 4 workers and keep using the workers so no reloading each time.

    # Added: I want to set up balanced_batches
    if args.balanced_batches:
        sampler = BalancedBatchSampler(training_dataset, args.batch_size)
        training_iterator = DataLoader(training_dataset, batch_sampler=sampler, num_workers=4, persistent_workers=True)
    else:
        training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)

    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)

    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    args.start_time = time.time()

    print (driving_policy)
    print (opt)
    print (args)

    args.class_dist = get_class_distribution(training_iterator, args)

    best_val_accuracy = 0
    for epoch in range(args.n_epochs):
        print ('EPOCH ', epoch)

        #
        # YOUR CODE GOES HERE
        #

        # Train the driving policy
        train_discrete(driving_policy, training_iterator, opt, args)
        # Evaluate the driving policy on the validation set
        val_acc = test_discrete(driving_policy, validation_iterator, opt, args)
        # If the accuracy on the validation set is a new high then save the network weights
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(driving_policy.state_dict(), args.weights_out_file)
            print('New best val acc: {:.2f}% -> saved to {}'.format(best_val_accuracy, args.weights_out_file))


    return driving_policy

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights",
                        required=True)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)
    parser.add_argument("--balanced_batches", type=str2bool, help="use mini batch contain both straight and turning samples evenly.", default=False)

    args = parser.parse_args()

    main(args)
