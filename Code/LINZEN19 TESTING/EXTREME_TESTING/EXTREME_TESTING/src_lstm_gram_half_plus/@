from agreement_acceptor import FullGramSentence
import filenames

pvn = FullGramSentence(filenames.deps, prop_train=0.8, output_filename='output_log.txt', len_after_verb=10)
pvn.pipeline(train=False,load=True,model="lstm_fullGram_100_50_emb.pkl", load_data=False, epochs=10, model_prefix='lstm_fullGram',test_size=7000, data_name='fullGram', test_external=True, load_external=True, external_file = filenames.external_file)

