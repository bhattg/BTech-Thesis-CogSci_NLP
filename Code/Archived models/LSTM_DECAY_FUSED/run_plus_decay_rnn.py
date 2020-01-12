from agreement_acceptor_joint import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10, hidden_dim=50, embedding_dim=50)

pvn.pipeline(train=True,model="decay_fullGram.pkl", load_data=True,epochs=10, model_prefix='decay_fullGram', data_name='fullGram', test_size=7000)
