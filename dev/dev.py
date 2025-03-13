
import os
from absl import flags, app
from planiverse.problems.real_world_problems.urban_planning.utils.config import Config
from planiverse.problems.real_world_problems.urban_planning.envs.city import CityEnv

sandboxdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandboxdev')
os.makedirs(sandboxdir, exist_ok=True)

cfg = Config("hlg", "111", sandboxdir, "root-dir", "gsca")

env = CityEnv(cfg)

pass
