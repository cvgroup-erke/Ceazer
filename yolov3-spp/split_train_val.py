import os
import random
# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('--input_label_path', type=str, help='input txt label path')
# parser.add_argument('--output_label_path', type=str, help='output txt label path')
# opt = parser.parse_args()

train_percent = 0.8
val_percent = 0.2
# test_percent = 0.1

# inputfilepath = opt.input_label_path
# txtsavepath = opt.output_label_path
input_path = r"D:\CV\Erke\data\labels"
out_path = r"D:\CV\Erke\data"
total_input = os.listdir(input_path)
if not os.path.exists(out_path):
  os.makedirs(out_path)

num=len(total_input)
print(num)
list=range(num)

# ftrainval = open(out_path + '/trainval.txt', 'w')
# ftest = open(out_path + '/test.txt', 'w')
ftrain = open(out_path + '/train.txt', 'w')
fval = open(out_path + '/val.txt', 'w')

for i in list:
    name=total_input[i][:-4]
    if (i+1) %5 == 0:
        fval.write(os.path.join(out_path, "images", name) + ".jpg" +'\n')
    # elif i % 11 == 0:
    #     ftest.write(os.path.join(out_path, "images", name) + ".jpg" +'\n')
    else:
        ftrain.write(os.path.join(out_path, "images", name) + ".jpg" +'\n')

ftrain.close()
fval.close()
# ftest.close()
