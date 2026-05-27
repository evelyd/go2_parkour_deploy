import os
import sys
import importlib.abc
import importlib.machinery
from unittest.mock import MagicMock

# Bunch of hacks to not have to import parkour_isaaclab
class MockPackage(MagicMock):
    __path__ = []

class CatchAllMockFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(('omni', 'isaacsim', 'pxr', 'carb', 'warp', 'mujoco.viewer')):
            return importlib.machinery.ModuleSpec(fullname, CatchAllMockLoader())
        return None

class CatchAllMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return MockPackage()

    def exec_module(self, module):
        pass

sys.meta_path.insert(0, CatchAllMockFinder())
# --------------------------------------------

from scripts.utils import load_local_cfg
from core.deployment_player import DeploymentPlayer
import numpy as np

def main(args):
    """Play with RSL-RL agent."""
    logs_path = f"/home/edelia-iit.local/git/IsaacLab/Isaaclab_Parkour/logs/rsl_rl/unitree_go2_parkour/{args.expid}"
    cfgs_path = os.path.join(logs_path, 'params')
    env_cfg = load_local_cfg(cfgs_path, 'env')
    agent_cfg = load_local_cfg(cfgs_path, 'agent')
    env_cfg.scene.num_envs = 1

    player = DeploymentPlayer(
        env_cfg=env_cfg,
        agent_cfg = agent_cfg,
        network_interface= args.interface,
        logs_path = logs_path,
    )

    player.reset(maximum_iteration = args.n_eval)

    save_info = {}
    # Force exactly 1000 steps of the simulation
    for step in range(1000):
        obs, terminated, timeout, extras, actions = player.play()

        # Go2 deploy, after player.play()
        d_go2 = extras["observations"]["depth_camera"].detach().cpu().numpy()

        save_info[step] = {
            'obs': obs.detach().cpu().numpy(),
            'actions': actions.detach().cpu().numpy(),
            'depth_camera': extras['observations']['depth_camera'].detach().cpu().numpy()
        }

        # Optional: Print every 100 steps so you know it's alive in the terminal
        if step % 100 == 0:
            print(f"[DEBUG] Executing step {step} / 1000")

        if terminated or timeout:
           print("[DEBUG] Episode ended, resetting...")
           player.reset(extras = extras)

    print('Eval Done')
    np.save('save_info.npy', save_info)
    # Also save a human-readable txt copy of the saved info
    open('save_info.txt', 'w').write(repr(save_info))

    if hasattr(player, 'env'):
        if hasattr(player.env, 'video_writer') and player.env.video_writer is not None:
            player.env.video_writer.close()
        else:
            print("\n[ERROR] video_writer was None. Frames were never captured.")

        player.env.close()

    sys.exit()

if __name__ == "__main__":
    import argparse
    # mp.set_start_method("spawn")
    parser = argparse.ArgumentParser(description='sim_2_sim')
    parser.add_argument("--rl_lib", type=str, default='rsl_rl')
    parser.add_argument("--task", type=str, default='unitree_go2_parkour')
    parser.add_argument("--expid", type=str, default='2025-09-03_12-07-56')
    parser.add_argument("--interface", type=str, default='lo')
    parser.add_argument("--use_joystick", action='store_true', default=False)
    parser.add_argument("--n_eval", type=int, default=10)
    args = parser.parse_args()
    main(args)
