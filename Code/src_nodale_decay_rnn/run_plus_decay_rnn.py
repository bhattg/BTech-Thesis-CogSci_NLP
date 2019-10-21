from agreement_acceptor_decay_rnn import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10)

pvn.pipeline(train=True,model="nodale_decay_fullGram.pkl", load_data=True,epochs=10, model_prefix='nodale_decay_fullGram', data_name='fullGram',load=False, test_size=7000)
