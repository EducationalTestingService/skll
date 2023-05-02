from os.path import abspath, dirname, join

_my_dir = abspath(dirname(__file__))
config_dir = join(_my_dir, "configs")
backward_compatibility_dir = join(_my_dir, "backward_compatibility")
examples_dir = join(dirname(_my_dir), "examples")
output_dir = join(_my_dir, "output")
other_dir = join(_my_dir, "other")
train_dir = join(_my_dir, "train")
test_dir = join(_my_dir, "test")
