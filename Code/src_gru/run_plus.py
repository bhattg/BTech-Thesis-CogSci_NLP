from agreement_acceptor import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, embedding_size=50, hidden_dim = 50)
pvn.pipeline(train=False,load=True,model="gru_fullGram.pkl", load_data=True, epochs=10, model_prefix='gru_fullGram',test_size=0, data_name='fullGram')
