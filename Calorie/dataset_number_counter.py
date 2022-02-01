import sys
import pickle

import os
def find_files(directory, suffix='.avi'):
    if not os.path.exists(directory):
        raise ValueError("Directory not found {}".format(directory))

    matches = []
    for root, dirnames, filenames in os.walk(directory):
        #print(filenames)
        for dirname in dirnames:
            for root, dirnames, filenames in os.walk(directory+dirname):
                #print(dirnames)
                for filename in filenames:
                    full_path = os.path.join(directory+dirname, filename)
                    #if filename.endswith(suffix):
                    matches.append(full_path)
    return matches
def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, dirnames, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

root = "/cvhci/data/activity/Calorie/calorie_common_dataset"

seq=recursive_glob(root, suffix=".npy")
final_list = []
video_list = []
#print(len(seq))
#sys.exit()
#for a,b,c in  os.walk(root):
#    print(a)
#    print(b)
#    print(c)
file_name = "ntu_staffs/ntu_calorie_all_skvideo.pkl"
print(len(seq))
"""
for file in seq:
    #print(file)
    #label_index = int(file.split('/')[-2][1:])
    label_index = int(file.split('/')[-1].split('A')[1].split('.skeleton')[0])
    name = int(file.split('/')[-1].split('.')[0][1:4])
    subject = int(file.split('/')[-1].split('P')[1].split('R')[0])
    #print(file)
    ##print(subject)
    #print(name)
    #print(label_index)
    #sys.exit()
    #if (label_index in [1, 7, 13, 19, 25, 31, 37, 43, 49]) and
    if (33<label_index<40) and (name in [1,2,3,4,5,6,7,8]):
        final_list.append(file)
"""