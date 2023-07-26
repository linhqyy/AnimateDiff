from glob import glob
import os


basedir                = os.getcwd()
init_images_dir        = os.path.join(basedir, "init_images")
test = glob(os.path.join(init_images_dir, "*"))

print(test)

