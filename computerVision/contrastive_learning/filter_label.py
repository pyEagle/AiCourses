import os
import sys
import shutil

in_dir = sys.argv[1]
out_dir = sys.argv[2]
for f in os.listdir(in_dir)
    fn = os.path.join(in_dir, f)
    gn = os.path.join(out_dir, f)
    with open(fn, 'r') as fid:
        flag = False
        for line in fid:
            items = line.split(' ')
            if items[0]=='5':
                flag = True
                break

        if not flag:
            shutil.copy(fn, gn)
          
