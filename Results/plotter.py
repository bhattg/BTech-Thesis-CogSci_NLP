import os
import matplotlib.pyplot as plt
import numpy as np
import math

#root = "C:\\Users\\Asus\\Dropbox\\BTP_Gantavya_Hritik\\experiments\\Raw_Results\\FULL GRAMAR\\CNN_LATEST_WITH_SAVED_BANSAL_DATA\\logs"
root = "D:\\Semester_7\\BTP\\BTech-Thesis-CognitionInLanguage\\Results\\"
masterDict={} #key is the number of intervening noun
	# val is the corresponding accuracy

def findMax(li):
	maxi=-1
	for j in li:
		if(j>maxi):
			maxi=j 
	return maxi

def plot_with_int_nouns():
	x_list=masterDict.keys()
	y_list_acc=[]
	y_list_num=[]
	for key in x_list:
		accList,num=zip(*masterDict[key])
		y_list_acc.append(findMax(accList)*100)
		y_list_num.append(math.log(num[-1]))
	return (list(x_list),y_list_acc,y_list_num)


def accuracy_plots_intervening_nouns(root,includeIntNoun=False):
	# this function creates accuracy plots for different 
	# number of intervening nouns along with the total
	# accuracy
	with open(os.path.join(root,"output_log_eirnn_pvn.txt")) as file:
		lines=file.readlines()
	checker={} #type dictionary
	accuracy_list=[]
	for line in lines:
		try:
			if(type(eval(line))==type(checker)):
				tmpDict=eval(line)
				correct=0
				total=0
				for key in tmpDict.keys():
					if key not in masterDict:
						masterDict[key]=[]
					if key in masterDict:
						masterDict[key].append(tmpDict[key])
					correct+=tmpDict[key][0]*tmpDict[key][1]
					total+=tmpDict[key][1]
				accuracy_list.append(correct/total)
	
		except (SyntaxError,NameError) as e:
			pass

	maxAcc=findMax(accuracy_list)
	print("Maximum Accuracy: " + str(maxAcc));

	if(includeIntNoun):
		x_,y_,z_=plot_with_int_nouns()
		zipped=zip(x_[:8],y_[:8],z_[:8])
		res=sorted(zipped,key = lambda x: x[0])
		#x_,y_,z_=zip(*res)
		d={0: (0.9279810582994694, 1146254), 2: (0.8422569743555762, 60364), 1: (0.8887360275150473, 180265), 3: (0.8020687587447531, 20012), 5: (0.7269949066213922, 2945), 4: (0.7638149815800246, 7329), 7: (0.6593220338983051, 590), 6: (0.7118367346938775, 1225), 8: (0.656934306569343, 274), 9: (0.6574074074074074, 108), 10: (0.625, 56), 11: (0.5, 34), 12: (0.8461538461538461, 13), 17: (1.0, 1), 13: (0.4, 10), 15: (0.5, 4), 16: (0.5, 2), 14: (1.0, 2), 18: (0.0, 2)}
		x_=[i for i in range(8)]
		y_=[96.38,93.56,90.39,86.84,82.89,79.08,75.92,69.32]
		z_=[]
		s=0
		t=0
		for h in x_:
			t+=d[h][1]
			s+=y_[h]*d[h][1]
		print(s/t)
		print(x_)
		print(y_)
		print(z_)

		# fig=plt.figure()
		# plt.subplot(2,1,1)
		plt.xlabel("Intervening Nouns")
		plt.ylabel("Accuracy(%)")
		plt.title("Accuracy of DecayRNN with number of intervening nouns")
		marks=[True for i in range(len(x_))]
		plt.plot(x_,y_,marker='o',color='r')

		# plt.subplot(2,1,2)
		# plt.xlabel("Number of Noun Attractors")
		# plt.ylabel("log scale")
		# plt.plot(x_,z_,marker='o',color='r')
		plt.grid()
		plt.show()
	else:
		accuracy_list=[100*i for i in accuracy_list]
		plt.plot(accuracy_list,marker='o',color='r')
		plt.title("Accuracy Plot for CNN")
		plt.xlabel("Training Size")
		plt.ylabel("Accuracy(%)")
		plt.show()


def accuracy_plots(root, tot_test): 
	with open(os.path.join(root, "output_log.txt")) as file:
		line = file.readlines()    
	accuracy_list= []
	for i in range(len(line)):
		tokens = line[i].split()
		if(tokens[0]=="Accuracy"):
			accuracy_list.append(int(tokens[-1]))
	ll = [100*(1-accuracy_list[i]/tot_test) for i in range(len(accuracy_list))]
	return ll


def grad_plots(root, num_parameter):
	with open(os.path.join(root, "grad_output_log.txt")) as file:
		line = file.readlines()
	grads = []
	for lines in line:
		tokens = lines.split()
		if tokens[0].isdigit():
			grads.append((tokens[-1]))
	tot = []
	i=0
	assert len(grads)%num_parameter == 0
	while i<len(grads):
		tot.append(np.asarray(grads[i:i+num_parameter]))
		i+=num_parameter
	grad_norms = [np.linalg.norm(grad_vec) for grad_vec in tot]
	return grad_norms

def alpha_plot(root):
	with open(os.path.join(root,"alpha_output_log.txt")) as file:
		lines=file.readlines()
	alpha_list=[]
	count=0
	for line in lines:
		if "tensor(" in line:
			count+=1
			tmp=line.split(",")
			tmp_=tmp[0].split("(")
			temp_alpha=eval(tmp_[1])
			if(count%500==0):
				alpha_list.append(temp_alpha)

	plt.plot(alpha_list)
	plt.ylabel("alpha")
	plt.xlabel("Training Length")
	plt.title("Alpha Value with time Vanilla Decay_RNN")
	plt.show()


def main():	
	accuracy_plots_intervening_nouns(root,True)
	#alpha_plot(root)

	

if __name__=="__main__":
	main()
