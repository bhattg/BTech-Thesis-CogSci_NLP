from agreement_acceptor_decay_rnn import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, batch_size=128,embedding_size=50,hidden_dim=50,prop_train=0.1, output_filename='output_log.txt', len_after_verb=10)
#Set the num_layers in decay_rnn.py
pvn.pipeline(train=True,batched=True,learning_rate=0.002,model="decay_fullGram.pkl",load_data=True,batch_size=128,epochs=40, model_prefix='decay_fullGram', data_name='fullGram',load=False, test_size=7000)


                
         
