import os
import matplotlib.pyplot as plt
import numpy as np
import math

#root = "C:\\Users\\Asus\\Dropbox\\BTP_Gantavya_Hritik\\experiments\\Raw_Results\\FULL GRAMAR\\CNN_LATEST_WITH_SAVED_BANSAL_DATA\\logs"
# root = "D:\\Semester_7\\BTP\\src_decay_rnn_half_gram_plus\\logs"
root="D:\\Semester_7\\BTP\\BTech-Thesis-CognitionInLanguage\\Results\\full data testing\\saccadeDecayRNN\\Sigmoid version\\logs"
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
	with open(os.path.join(root,"testing_result_output_log.txt")) as file:
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
		zipped=zip(x_,y_,z_)
		res=sorted(zipped,key = lambda x: x[0])
		x_,y_,z_=zip(*res)
		print(x_)
		print(y_)
		print(z_)

		fig=plt.figure()
		plt.subplot(2,1,1)
		plt.xlabel("Noun Attractors")
		plt.ylabel("Accuracy")
		plt.title("Accuracy of Decay_RNN with number of intervening nouns")
		marks=[True for i in range(len(x_))]
		plt.plot(x_,y_,marker='o',color='b')

		for x,y in zip(x_,y_):
			label = "{:.2f}".format(y)
			plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


		plt.subplot(2,1,2)
		plt.xlabel("Number of Noun Attractors")
		plt.ylabel("log scale")
		plt.title("Number of examples in test set")
		plt.plot(x_,z_,marker='o',color='b')

		for x,y in zip(x_,z_):
			label = "{:.2f}".format(y)
			plt.annotate(label, # this is the text
                 (x,y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,5), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

		plt.show()
	else:
		accuracy_list=[100*i for i in accuracy_list]
		plt.plot(accuracy_list,marker='o',color='b')
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
