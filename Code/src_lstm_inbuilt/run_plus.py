from agreement_acceptor import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)
pvn.pipeline(train=True, load_data=True, epochs=10, model_prefix='lstm_fullGram',test_size=7000, data_name='fullGram')
