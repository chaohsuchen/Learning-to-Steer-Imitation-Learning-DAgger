import train_policy
import racer
import argparse
import os
import matplotlib.pyplot as plt
from utils import str2bool

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", type=int, default=10)
    parser.add_argument("--weighted_loss", type=str2bool, help="use inverse-frequency class weights", default=False)
    parser.add_argument("--balanced_batches", type=str2bool, help="use balanced mini-batches", default=False)
    args = parser.parse_args()

    #####
    ## Enter your DAgger code here
    ## Reuse functions in racer.py and train_policy.py
    ## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where
    #####
    # Added racer.run() required params!
    args.out_dir = args.train_dir
    args.save_expert_actions = True
    args.expert_drives = False
    args.timesteps = 100000

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs('./weights', exist_ok=True)

    avg_ctes, ep_lengths = [], []
    print ('TRAINING LEARNER ON INITIAL DATASET')
    args.weights_out_file = './weights/learner_0.weights'
    learner = train_policy.main(args)
    # Use the one with the best val acc
    learner.load_weights_from(args.weights_out_file)

    for i in range(1, args.dagger_iterations+1):
        print ('GETTING EXPERT DEMONSTRATIONS (iteration %d)' % i)
        args.run_id = i
        avg_cte, ep_len = racer.run(learner, args)
        avg_ctes.append(avg_cte)
        ep_lengths.append(ep_len)


        print ('RETRAINING LEARNER ON AGGREGATED DATASET')
        args.weights_out_file = './weights/learner_%d.weights' % i
        learner = train_policy.main(args)
        learner.load_weights_from(args.weights_out_file)

    # Evaluate learner_10 once, don't save
    args.save_expert_actions = False
    avg_cte, ep_len = racer.run(learner, args)
    avg_ctes.append(avg_cte)
    ep_lengths.append(ep_len)

    # draw the plot!
    iters = list(range(len(avg_ctes)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iters, avg_ctes, marker='o', color='tab:blue')
    axes[0].set_xlabel('DAgger iteration')
    axes[0].set_ylabel('Per-timestep avg cross-track error')
    axes[1].plot(iters, ep_lengths, marker='o', color='tab:orange')
    axes[1].set_xlabel('DAgger iteration')
    axes[1].set_ylabel('Episode length (timesteps)')
    plt.tight_layout()
    plt.savefig('dagger_iterations.png')
    print(avg_ctes, ep_lengths)
