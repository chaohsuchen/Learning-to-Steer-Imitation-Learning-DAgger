"""Record an actual CNN-steered rollout as a looping GIF, without a display."""

import argparse
import os
from pathlib import Path
import sys

# Set these before importing the simulator / pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
import numpy as np
import torch

from driving_policy import DiscreteDrivingPolicy
from full_state_car_racing_env import FullStateCarRacingEnv
from car_racing import FPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path,
                        default=ROOT / "final_weights/learner_10.weights")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "images/driving-demo.gif")
    parser.add_argument("--timesteps", type=int, default=1500)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.timesteps < 1 or args.frame_stride < 1:
        parser.error("timesteps and frame-stride must be positive")

    # CPU loading keeps the included checkpoint usable without an Apple GPU.
    torch.set_num_threads(1)
    model = DiscreteDrivingPolicy(n_classes=20).to("cpu")
    model.load_state_dict(torch.load(args.weights, map_location="cpu", weights_only=True))
    # This course model defines eval(state, device), so use train(False) here.
    model.train(False)

    env = FullStateCarRacingEnv(seed=args.seed)
    frames = []
    errors = []
    # Match racer.py: Box2D's bindings reject NumPy float32 action scalars.
    action = np.zeros(3, dtype=np.float64)
    try:
        env.reset()
        with torch.inference_mode():
            for step in range(args.timesteps):
                state, expert, _, done, info = env.step(action)
                errors.append(info["cross_track_error"])
                if step % args.frame_stride == 0:
                    # Preserve the simulator's 1200 x 1000 display aspect ratio.
                    frames.append(env._create_image_array(env.surf, (600, 500)).copy())
                if done:
                    break
                action[0] = model.eval(state, device="cpu")
                action[1:] = expert[1:]
    finally:
        env.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, loop=0,
                   duration=1000 * args.frame_stride / FPS)
    print(f"Saved {args.output}: {len(frames)} frames, {len(errors)} simulation steps, "
          f"mean cross-track error {np.mean(errors):.4f}")


if __name__ == "__main__":
    main()
