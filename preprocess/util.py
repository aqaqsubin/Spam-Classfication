
import os
import errno

from shutil import rmtree

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
        
def del_folder(path):
    try:
        rmtree(path)
    except:
        pass