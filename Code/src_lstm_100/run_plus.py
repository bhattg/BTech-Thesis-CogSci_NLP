from agreement_acceptor import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.8,hidden_dim=100,embedding_size=100, output_filename='output_log.txt', len_after_verb=10,embedding_size=50, hidden_dim = 50)
pvn.pipeline(train=True,load=False,model="lstm_fullGram_100.pkl", load_data=True, epochs=10, model_prefix='lstm_fullGram_100',test_size=7000, data_name='fullGram')
