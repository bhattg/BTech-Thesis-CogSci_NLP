import pandas as pd 
import pandas as pd
import ast
import os
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
def draw_waveform_and_save(index, df1, df2, df3, path=None):
    sentence = df1["Input sentence"][index].split()
    l1 = ast.literal_eval(df1["Time stamp outputs"][index])
    l2 = ast.literal_eval(df2["Time stamp outputs"][index])
    l3 = ast.literal_eval(df3["Time stamp outputs"][index])
    for i in range(len(sentence)):
        sentence[i]=sentence[i]+"_"+str(i)
    plt.plot(sentence, l1, label="dale")
    plt.plot(sentence, l2, label="nodale")
    plt.plot(sentence, l3, label="lstm")
    plt.legend(loc="upper left")
    plt.xticks(rotation=90)
    if path==None:
        plt.show()
    else:
        plt.savefig(os.path.join(path, str(index)))
        plt.close()

root = "../file under test for waveform/"

lstm_dir = os.path.join(root, "lstm")
dale_dir = os.path.join(root, "dale")
nodale_dir = os.path.join(root, "nodale")

f1 = (os.listdir(lstm_dir))
f2 = (os.listdir(dale_dir))
f3 = (os.listdir(nodale_dir))
f1.sort()
f2.sort()
f3.sort()

print("Started Parsing tests!!!")

for i in tqdm(range(len(f1))):
	# for each kind of test, give the graphs. 
	loc1 = os.path.join(lstm_dir, f1[i])
	loc2 = os.path.join(dale_dir, f2[i])
	loc3 = os.path.join(nodale_dir, f3[i])
	assert f1[i] ==  f2[i] == f3[i]
	df1 = pd.read_csv(loc1).to_dict()
	df2 = pd.read_csv(loc2).to_dict()
	df3 = pd.read_csv(loc3).to_dict()
	assert len(df1["Input sentence"].keys()) == len(df2["Input sentence"].keys()) == len(df3["Input sentence"].keys())
	total_sentences = len(df1["Input sentence"].keys())
	if not os.path.isdir(os.path.join(root, f1[i])):
		os.mkdir(os.path.join(root, f1[i]))
	count=0
	dump_root = os.path.join(root, f1[i]) 
	for i in tqdm(range(total_sentences)):
	    l1 = np.asarray(ast.literal_eval(df1["Time stamp outputs"][i]))
	    l2 = np.asarray(ast.literal_eval(df2["Time stamp outputs"][i]))
	    l3 = np.asarray(ast.literal_eval(df3["Time stamp outputs"][i]))
	    if np.sum(l1!=l2)!=0 or np.sum(l3!=l2)!=0 or np.sum(l1!=l3)!=0:
	        # now we need to draw a waveform. 
	        draw_waveform_and_save(i, df1, df2, df3, path=dump_root)
	    else:
	        count+=1
	print("Sentences with exact same waveform were {}".format(count))





	