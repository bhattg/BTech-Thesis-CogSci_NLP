from agreement_acceptor_CNN import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, decay_vector=[0,0,0,0,0,0,0,2,5])
pvn.pipeline(train=True,batched=True, test_size=200000, batch_size=64, load=False, epochs=20, model_prefix='cnn_plus10', data_name='plus10')
