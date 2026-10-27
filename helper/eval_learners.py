import argparse
import matplotlib.pyplot as plt
import racer
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE

if __name__ == "__main__":
    args = argparse.Namespace(timesteps=100000, expert_drives=False,
                              save_expert_actions=False, out_dir='', run_id=0)
    avg_ctes, ep_lengths = [], []

    for i in range(11):
        print('Evaluating learner_%d' % i)
        net = DiscreteDrivingPolicy(n_classes=20).to(DEVICE)
        net.load_weights_from('../weights/learner_%d.weights' % i)
        avg_cte, ep_len = racer.run(net, args)
        avg_ctes.append(avg_cte)
        ep_lengths.append(ep_len)

    iters = list(range(len(avg_ctes)))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(iters, avg_ctes, marker='o', color='tab:blue')
    axes[0].set_xlabel('DAgger iteration')
    axes[0].set_ylabel('Per-timestep avg cross-track error')
    axes[1].plot(iters, ep_lengths, marker='o', color='tab:orange')
    axes[1].set_xlabel('DAgger iteration')
    axes[1].set_ylabel('Episode length (timesteps)')
    plt.tight_layout()
    plt.savefig('h_dagger_iterations_eval.png')
    print(avg_ctes, ep_lengths)
