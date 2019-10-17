from agreement_acceptor import GramHalfPlusSentence
import filenames

pvn = GramHalfPlusSentence(filenames.deps, prop_train=0.1, output_filename='output_log.txt', len_after_verb=10)
pvn.pipeline(train=True, load=True, epochs=10, model_prefix='lstm_ghp_plus10',test_size=50000, data_name='plus10', model="lstm_ghp_plus10.pkl")
