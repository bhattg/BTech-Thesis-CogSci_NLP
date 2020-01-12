import json
import multiprocessing
import os
import sys
import os.path as op
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import filenames
from utils import deps_from_tsv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BatchedDataset(Dataset):
    '''
    This class make a general dataset that we will use to generate 
    the batched training data
    '''
    def __init__(self, x_train, y_train):
        super(BatchedDataset, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        assert (x_train).shape[0] == (y_train).shape[0] 
        self.length =  (x_train).shape[0]
    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.length


class DECAY_RNN_Model(object):

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
            key = (deps_test[i]['n_intervening'], deps_test[i]['n_diff_intervening'])
            if not key in testing_dict.keys():
                testing_dict[key]=[]
            testing_dict[key].append((X_test[i], Y_test[i]))

        self.testing_dict=testing_dict

    serialized_attributes = ['vocab_to_ints', 'ints_to_vocab', 'filename',
                             'X_train', 'Y_train', 'deps_train',
                             'X_test', 'Y_test', 'deps_test']


    def __init__(self, filename=None, serialization_dir=None,
                 batch_size=128, embedding_size=50, hidden_dim = 50,
                 maxlen=50, prop_train=0.9, rnn_output_size=10,
                 mode='infreq_pos', vocab_file=filenames.vocab_file,
                 equalize_classes=False, criterion=None, len_after_verb=0,
                 verbose=1, output_filename='default.txt', batched=True, testing_batch_size=256):
        '''
        filename: TSV file with positive examples, or None if unserializing
        criterion: dependencies that don't meet this criterion are excluded
            (set to None to keep all dependencies)
        verbose: passed to Keras (0 = no, 1 = progress bar, 2 = line per epoch)
        '''
        self.testing_batch_size=testing_batch_size
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
        self.batched=batched
        # self.set_serialization_dir(serialization_dir)

    def log(self, message):
        with open('logs/' + self.output_filename, 'a') as file:
            file.write(str(message) + '\n')

    def log_debugger(self, message):
        with open('logs/debugger_' + self.output_filename, 'a') as file:
            file.write(str(message) + '\n')


    def log_grad(self, message):
        with open('logs/grad_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def log_alpha(self,message):
        with open('logs/alpha_' + self.output_filename, 'a') as file:
            file.write(message + '\n')

    def pipeline(self, train = True, batched=True, shuffle = True, num_workers= 0,
                 load = False, model = '', test_size=7000, 
                 train_size=None, model_prefix='__', epochs=20, data_name='Not', 
                 activation=False, df_name='_verbose_.pkl', load_data=False,learning_rate=0.01, 
                 save_data=False):

        self.batched= batched
        if (load_data):
            self.load_train_and_test(test_size, data_name)
        else :
            self.log('creating data from input data file {}'.format(str(self.filename)))
            examples = self.load_examples(data_name, save_data, None if train_size is None else train_size*10)
            self.create_train_and_test(examples, test_size, data_name, save_data)
        
        if batched:
            self.create_model_batched()
        else:
            self.create_model()
        if (load) :
            self.load_model(model)
        if (train) :
            self.train_batched(epochs, model_prefix, shuffle=shuffle, num_workers=num_workers,learning_rate=learning_rate)
        else:
            result_dict= self.test_model()
            print(result_dict)
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
                if token not in self.vocab_to_ints:
                    # zero is for pad
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
        self.model = torch.load(model).to(device)        

            

    def train_batched(self, n_epochs=10, model_prefix="__", shuffle=True, learning_rate=0.01, num_workers=0, test_after_every=100, annealing=False):
        self.log('Training Batched')
        if not hasattr(self, 'model'):
            self.create_model_batched()

##############################################################################
################### OPTIMIZER RELATED  @ AUTHOR gantavya #####################
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)
        patience = 5# used to reschedule the learning rate. 
        patience_counter = 0
        factor = 0.8
        threshold = 0.001
        max_acc = 0
##############################################################################
##############################################################################

        self.log(len(self.X_train))

        '''
        Since our Dataset class needs the array as the input and it is actually better to use array as the inputs, 
        so we will conver the training data to array
        '''
        
        total_batches = int(len(self.X_train)/self.batch_size)   # if len(x_train)%batch_size != 0 then total_batches +=1 (add one to the calc batches)
        x_train = np.asarray(self.X_train,dtype=int)   
        y_train = np.asarray(self.Y_train,dtype=int)

        if device == torch.device("cuda"):
            self.log("Training on NVIDIA GPUs")
            print("Training on NVIDIA GPUs")
        else:
            self.log("Training on CPU")
            print("Training on CPU")

        print("Total Train epochs : "+str(n_epochs))
        print("Total Train batches : "+str(total_batches))
        self.log("Total Train epochs : "+str(n_epochs))
        self.log("Total Train batches : "+str(total_batches))

        max_acc= 0

        #creating batches --- our batchify 
        # batch size everywhere should be taken as the self.batch size provided in the init 

        new_BatchedDataset =  BatchedDataset(x_train, y_train)
        DataGenerator =  DataLoader(new_BatchedDataset, batch_size= self.batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
        
        self.log("Started Training Phase !!")
        print("Started Training Phase !!")
        
        for epoch in range(n_epochs) :
            
            self.log('epoch : ' + str(epoch))
            self.log_grad('epoch : ' + str(epoch))
            batches_processed = 0
            batch_list=[]
            for x_batch, y_batch in DataGenerator :
                m = x_batch.shape[0]
                assert m == y_batch.shape[0]
                x_batch = x_batch.view(m, self.maxlen).to(device)
                y_batch = y_batch.view(m).to(device)
                batch_list.append((x_batch, y_batch))

                if batches_processed!=0 and batches_processed%test_after_every==0 :
                    acc =  self.results_batched()
                    if (acc >= max_acc) :
                        model_name = model_prefix + '.pkl'
                        torch.save(self.model, model_name)
                        max_acc = acc   
                    else :
                        if annealing:
                            if patience_counter>=patience:
                                # reschedule the learning rate 
                                for g in optimizer.param_groups:
                                    g['lr'] = factor*g['lr']
                                    new_lr = g['lr']
                                    if new_lr < threshold:
                                        g['lr'] = threshold
                                        new_lr= threshold
                                patience_counter=0
                                print("Re-Scheduling learning rate to {}".format(new_lr))
                                self.log("Rescheduling learning rate to {}".format(new_lr))

                            else:
                                patience_counter+=1
                        


                for name,param in self.model.named_parameters():
                    if(name=="cell_0.rgate"):
                        self.log_alpha(str(param))                 

                self.model.zero_grad()
                output, _ , _  = self.model(x_batch)
                loss = loss_function(output,y_batch)
                loss.backward(retain_graph=True)
                optimizer.step()
                batches_processed+=1
                self.log('batches processed : ' + str(batches_processed))

                counter = 0
                self.log_grad('batches processed : ' + str(batches_processed))
                for param in self.model.parameters():
                    if param.grad is not None:
                        # print(counter, param.shape)
                        self.log_grad(str(counter) + ' : ' + str(param.grad.norm().item()))
                        counter += 1

            acc = self.results_batched()
            if (acc > max_acc) :
                model_name = model_prefix + '.pkl'
                torch.save(self.model, model_name)
                max_acc = acc

