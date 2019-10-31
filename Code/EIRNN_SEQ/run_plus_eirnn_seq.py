from agreement_acceptor_eirnn_seq import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.8, output_filename='output_log.txt', len_after_verb=10, hidden_dim=50, embedding_size=50)

pvn.pipeline(train=True,model="eirnn_seq.pkl", load_data=True,load=False,epochs=10, model_prefix='eirnn_seq_fullgram', data_name='fullGram', test_size=7000, test_external=False, load_external=False, external_file=None)
