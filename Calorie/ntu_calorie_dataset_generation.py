import torch
from torch.utils.data import random_split
import pickle
from sklearn.model_selection import train_test_split

path = 'ntu_staffs/ntu_calorie_train_test_skvideo_re.pkl'
f=open(path,'rb')
dataset=pickle.load(f)
print(len(dataset))
#dataset = range(10)
save_path_train_x = '/home/kpeng/calorie/MUSDL/Calorie/ntu_staffs/ntu_calorie_train_x_re.pkl'
save_path_train_y = '/home/kpeng/calorie/MUSDL/Calorie/ntu_staffs/ntu_calorie_train_y_re.pkl'
save_path_test_x = '/home/kpeng/calorie/MUSDL/Calorie/ntu_staffs/ntu_calorie_test_x_re.pkl'
save_path_test_y = '/home/kpeng/calorie/MUSDL/Calorie/ntu_staffs/ntu_calorie_test_y_re.pkl'
label= []
for file in dataset:
    label.append(int(file.split('/')[-1].split('A')[-1].split('_')[0]))
#train_dataset, test_dataset = random_split(
#    dataset=dataset,
#    lengths=[2783, 1192]
#)
x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size=0.3, random_state=42)
#print(y_train)
#print(y_test)
print(len(x_train))
print(len(x_test))
with open(save_path_train_x, 'wb') as fp:
    pickle.dump(x_train, fp)
with open(save_path_train_y, 'wb') as fp:
    pickle.dump(y_train, fp)
with open(save_path_test_x, 'wb') as fp:
    pickle.dump(x_test, fp)
with open(save_path_test_y, 'wb') as fp:
    pickle.dump(y_test, fp)

