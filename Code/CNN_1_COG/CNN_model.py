import json
import sys
import multiprocessing
import os
import os.path as op
import random
import torch
# import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pickle
from Cog_CNN import Cognitive_CNN
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import filenames
from utils import deps_from_tsv,dump_to_csv, dump_dict_to_csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
device = cpu

class BatchedDataset(Dataset):
    '''
    This class make a general dataset that we will use to generate 
    the batched training data
    '''
    def __init__(self, x_train, y_train):
        super(BatchedDataset, self).__init__()
        self.x_train = np.asarray(x_train)
        self.y_train = np.asarray(y_train)
        assert (self.x_train).shape[0] == (self.y_train).shape[0] 
        self.length =  (self.x_train).shape[0]
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.length




class CNNModel(object):

    serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
                             'X_train', 'Y_train', 'deps_train',
                             'X_test', 'Y_test', 'deps_test']

    def input_to_string(self, x_input):
        #x_input is the example we want to convert to the string 
        #x_input should be in the form of 1D list. 
        example_string = ""
        for token in x_input:
            if token == 0:
                continue
            str_tok =  self.ints_to_vocab[token]
            example_string+=str_tok+" "
        return example_string



    def demark_testing(self):
        X_test=self.X_test
        Y_test=self.Y_test
        deps_test=self.deps_test

        testing_dict={}


        assert len(X_test)==len(Y_test) and len(Y_test)==len(deps_test)
        for i in (range(len(X_test))):
            key = deps_test[i]['n_intervening']
            if not key in testing_dict.keys():
                testing_dict[key]=[]
            testing_dict[key].append((X_test[i], Y_test[i]))

        for key in testing_dict.keys():
            x_ , y_  = zip(*testing_dict[key])
            x_ = list(x_)
            y_ = list(y_)
            testing_dict[key]= DataLoader(BatchedDataset(x_, y_),  batch_size= 64, shuffle=True, num_workers=0, drop_last=False)
        self.testing_dict=testing_dict
        



    def __init__(self, filename=None, serialization_dir=None,
                 batch_size=1, embedding_size=50, hidden_dim = 50,
                 maxlen=50, prop_train=0.9, rnn_output_size=10,
                 mode='infreq_pos', vocab_file=filenames.vocab_file,
                 equalize_classes=False, criterion=None, len_after_verb=0,
                 verbose=1, output_filename='default.txt', decay_vector=[]):
        '''
        filename: TSV file with positive examples, or None if unserializing
        criterion: dependencies that don't meet this criterion are excluded
            (set to None to keep all dependencies)
        verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
        '''
        self.filename = filename
        self.vocab_file = vocab_file
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.prop_train = prop_train
        self.mode = mode
        self.rnn_output_size = rnn_output_size
        self.maxlen = maxlen
        self.equalize_classes = equalize_classes
        self.criterion = (lambda x: True) if criterion is None else criterion
        self.len_after_verb = len_after_verb
        self.verbose = verbose
        self.output_filename = output_filename
        # self.set_serialization_dir(serialization_dir)

    def log_t(self, message):
        with open('logs/tr_' + self.output_filename, 'a') as file:
            file.write(str(message) + '\n')
    def log(self, message):
        with open('logs/' + self.output_filename, 'a') as file:
            file.write(str(message) + '\n')

    def log_predicted(self, message):
        with open('logs/grad_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_grad(self, message):
        with open('logs/grad_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_loss(self, message):
         with open('logs/loss_' + self.output_filename, 'a') as file:
            file.write(message + '\n')       

    def create_CNN_model(self):
        self.log('Creating model')
        self.log('vocab size : ' + str(len(self.vocab_to_ints)))
        self.model = Cognitive_CNN(self.decay_vector, vocab_size = len(self.vocab_to_ints)+1)

    def plot_results(self, result_dict):
        ordinate = [] 
        abscissa = []
        for key in result_dict.keys():
            ordinate.append(result_dict[key])
            abscissa.append(key)
        plt.plot(abscissa, ordinate)
        plt.show()


    def pipeline(self, train = True, batched=False, batch_size = 32, shuffle = True, num_workers= 0,
                 load = False, model = '', test_size=200000, 
                 train_size=None, model_prefix='__', epochs=20, data_name='Not', 
                 activation=False, df_name='_verbose_.pkl', load_data=False, 
                 save_data=False):
        self.batched= batched
        if (load_data):
            self.load_train_and_test(test_size, data_name)
        else :
            self.log('creating data')
            examples = self.load_examples(data_name, save_data, None if train_size is None else train_size*10)
            self.create_train_and_test(examples, test_size, data_name, save_data)

        if (load) :
            print("LOADED MODEL!")
            self.load_model(model)
        else:
            self.create_CNN_model()


        print("Pre train steps complete!") 

        if (train) :
            self.train_batched(epochs, model_prefix, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        else:
            result_dict= self.test_model()
            self.cross_validate(0)
            print(result_dict)
            # self.plot_results(result_dict)

        


        print('Data : ',  data_name)
        self.log(data_name)



    def load_examples(self,data_name='Not',save_data=False, n_examples=None):
        '''
        Set n_examples to some positive integer to only load (up to) that 
        number of examples
        '''
        self.log('Loading examples')
        if self.filename is None:
            raise ValueError('Filename argument to constructor can\'t be None')

        self.vocab_to_ints = {}
        self.ints_to_vocab = {}
        examples = []
        n = 0

        deps = deps_from_tsv(self.filename, limit=n_examples)

        for dep in deps:
            tokens = dep['sentence'].split()
            if len(tokens) > self.maxlen or not self.criterion(dep):
                continue

            tokens = self.process_single_dependency(dep)
            ints = []


            for token in tokens:
                if token not in self.vocab_to_ints:                                         #save the vocab to int dict  
                    # zero is for pad                                                       #save the int to vocab dict                                                                                                     
                    x = self.vocab_to_ints[token] = len(self.vocab_to_ints) + 1
                    self.ints_to_vocab[x] = token
                ints.append(self.vocab_to_ints[token])




            examples.append((self.class_to_code[dep['label']], ints, dep))
            n += 1
            if n_examples is not None and n >= n_examples:
                break

        if (save_data) :
            with open('plus5_v2i.pkl', 'wb') as f:
                pickle.dump(self.vocab_to_ints, f)
            with open('plus5_i2v.pkl', 'wb') as f:
                pickle.dump(self.ints_to_vocab, f)

        return examples

    def load_model(self, model) :
        self.model = torch.load(model)

    def train_validate(self, batches_processed):

        total_examples=0
        acc=0
        with torch.no_grad():
            for (x_batch, y_batch) in batches_processed:
                predicted = self.model.predict(x_batch.to(torch.long))
                total_examples+=y_batch.shape[0]
                acc += np.sum(y_batch.numpy()==predicted.squeeze().numpy())
            self.log_t("Training accuracy on {} examples  is {} ".format(total_examples, acc/total_examples))
            print("Training accuracy on {} examples  is {} ".format(total_examples, acc/total_examples))


    def train_batched(self, n_epochs=10, model_prefix="__", batch_size=32, shuffle=True, learning_rate=0.005, num_workers=0):
        self.log('Training Batched')
        if not hasattr(self, 'model'):
            self.create_CNN_model()
        
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate,amsgrad=True)
        prev_param = list(self.model.parameters())[0].clone()
        max_acc = 0
        self.log(len(self.X_train))

        total_batches = int(len(self.X_train)/batch_size)
        x_train = np.asarray(self.X_train)
        y_train = np.asarray(self.Y_train)

        print("Num Epochs : "+str(n_epochs))
        print("Total Train batches : "+str(total_batches))

        new_BatchedDataset =  BatchedDataset(x_train, y_train)
        DataGenerator =  DataLoader(new_BatchedDataset, batch_size= batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
        ind=0
        for epoch in range(n_epochs) :
            
            self.log('epoch : ' + str(epoch))
            self.log_grad('epoch : ' + str(epoch))
            batches_processed = 0
            batch_list=[]
            for x_batch, y_batch in DataGenerator :
                batch_list.append((x_batch, y_batch))
                if (batches_processed+1)%10==0:
                    self.log("{}/{} Batches Processed".format(batches_processed, total_batches))
                    self.log_loss("Loss : "+str(loss))
                    self.train_validate(batch_list)
                    acc =  self.cross_validate(batches_processed)
                    if (acc >= max_acc) :
                        model_name = model_prefix +str(ind)+ '.pkl'
                        torch.save(self.model, model_name)
                        max_acc = acc              
                        ind+=1      
                    _ =  self.test_model()



                self.model.zero_grad()
                output = self.model(x_batch.to(torch.long))
                loss = loss_function(output.squeeze(),y_batch.to(torch.float)) + self.model.norm_penalty()
                loss.backward(retain_graph=True)
                optimizer.step()
                batches_processed+=1

                counter = 0
                self.log_grad('batches processed : ' + str(batches_processed))
                grads_LL = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        # print(counter, param.shape)
                        gradient = param.grad.norm().item()
                        grads_LL.append(gradient)
                        self.log_grad(str(counter) + ' : ' + str(gradient))
                        counter += 1
                self.log_grad("absolute derivative : {}".format(np.linalg.norm(np.asarray(grads_LL))))

            acc = self.cross_validate(batches_processed)
            if (acc > max_acc) :
                model_name = model_prefix + '.pkl'
                torch.save(self.model, model_name)
                max_acc = acc
            

            
  
