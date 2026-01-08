from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

# Explicitly import VPL trainers to ensure they are registered
try:
    from federatedscope.llm.trainer import vpl_reward_choice_trainer
    from federatedscope.llm.trainer import vpl_reward_trainer
except ImportError:
    pass
