
import shutil, os

path_a = '../images/set_a_imgs/'

for file in os.listdir(path_a):
    if "artifact" in file:
        if not os.path.exists('../images/train/artifact/'):
            os.makedirs('../images/train/artifact/')
        shutil.move(path_a+file, '../images/train/artifact/')
    if "normal" in file:
        if not os.path.exists('../images/train/normal/'):
            os.makedirs('../images/train/normal/')
        shutil.move(path_a+file, '../images/train/normal/')
    if "murmur" in file:
        if not os.path.exists('../images/train/murmur/'):
            os.makedirs('../images/train/murmur/')
        shutil.move(path_a+file, '../images/train/murmur/')
    if "extrahls" in file:
        if not os.path.exists('../images/train/extrahls/'):
            os.makedirs('../images/train/extrahls/')
        shutil.move(path_a+file, '../images/train/extrahls/')
    if "extrastole" in file:
        if not os.path.exists('../images/train/extrastole/'):
            os.makedirs('../images/train/extrastole/')
        shutil.move(path_a+file, '../images/train/extrastole/')
    if "unlabelled" in file:
        if not os.path.exists('../images/test/'):
            os.makedirs('../images/test/')
        shutil.move(path_a+file, '../images/test/')
    else:
        pass

path_b = '../images/set_b_imgs/'

for file in os.listdir(path_b):
    if "artifact" in file:
        if not os.path.exists('../images/train/artifact/'):
            os.makedirs('../images/train/artifact/')
        shutil.move(path_b+file, '../images/train/artifact/')
    if "normal" in file:
        if not os.path.exists('../images/train/normal/'):
            os.makedirs('../images/train/normal/')
        shutil.move(path_b+file, '../images/train/normal/')
    if "murmur" in file:
        if not os.path.exists('../images/train/murmur/'):
            os.makedirs('../images/train/murmur/')
        shutil.move(path_b+file, '../images/train/murmur/')
    if "extrahls" in file:
        if not os.path.exists('../images/train/extrahls/'):
            os.makedirs('../images/train/extrahls/')
        shutil.move(path_b+file, '../images/train/extrahls/')
    if "extrastole" in file:
        if not os.path.exists('../images/train/extrastole/'):
            os.makedirs('../images/train/extrastole/')
        shutil.move(path_b+file, '../images/train/extrastole/')
    if "unlabelled" in file:
        if not os.path.exists('../images/test/'):
            os.makedirs('../images/test/')
        shutil.move(path_b+file, '../images/test/')
    else:
        pass