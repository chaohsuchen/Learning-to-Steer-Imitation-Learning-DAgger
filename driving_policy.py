import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class DiscreteDrivingPolicy(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes


        self.features = nn.Sequential(
            #
            # YOUR CODE GOES HERE
            # Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
            #
            # output
            #  = floor((input + 2*padding - kernel_size) / stride) + 1
            nn.Conv2d(3, 24, 4, 2, 1),  # (128 + 2 - 4) / 2 + 1 = 64
            nn.ReLU(),
            nn.Conv2d(24, 36, 4, 2, 1), # (64 - 2) / 2 + 1 = 32
            nn.ReLU(),
            nn.Conv2d(36, 48, 4, 2, 1), # (32 - 2) / 2 + 1 = 16
            nn.ReLU(),
            nn.Conv2d(48, 64, 4, 2, 1), # 8
            nn.ReLU(),
            Flatten(),
        )

        self.classifier = nn.Sequential(
            #
            # YOUR CODE GOES HERE
            #
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            # This is wrong! shouldn't use nn.Flatten(64, self.n_classes)
            # Should use Linear as FC
            nn.Linear(64, n_classes)
        )

        self.apply(weights_init)


    def forward(self, x):
        f = self.features(x)
        logits = self.classifier(f)
        return logits

    def eval(self, state, device):
        state = state.astype(np.float32) / 255.0
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0) # we add this dim to feed into our model, so remember to pick [0] at the end.
        logits = self.forward(state)

        y_pred = logits.view(-1, self.n_classes)
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()

        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0

        #  File "/Users/victorchen/Desktop/master1/Robots/hw1/A1/racer.py", line 36, in run
        #  learner_action[0] = steering_network.eval(state, device=DEVICE)
        #     ~~~~~~~~~~~~~~^^^
        # ValueError: setting an array element with a sequence.
        # return steering_cmd
        return steering_cmd[0] # I modify to this version

    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)
