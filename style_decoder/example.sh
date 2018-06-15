
##########
The first example shows how to build the democratic generator.
1. We first translate the democratic data from English to French
2. We then train the democratic style generator
#########

# Translate the democratic data from English to French
# Note: Use onmt.Translator when using the English-French translation system
python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/political_data/democratic_only.train.en -output ../data/political_data/democratic_only.train.fr -replace_unk $true
python translate.py -model ../models/translation/english_french/english_french.pt -src ../data/political_data/democratic_only.dev.en -output ../data/political_data/democratic_only.dev.fr -replace_unk $true

# Preprocess democratic french source and democratic english target data for the generator
python preprocess.py -train_src ../data/political_data/democratic_only.train.fr -train_tgt ../data/political_data/democratic_only.train.en -valid_src ../data/political_data/democratic_only.dev.fr -valid_tgt ../data/political_data/democratic_only.dev.en -save_data data/democratic_generator -src_vocab ../models/translation/french_english/french_english_vocab.src.dict -tgt_vocab ../models/classifier/political_classifier/political_classifier_vocab.src.dict -seq_len 50

# Train the democratic style generator
python train_decoder.py -data data/democratic_generator.train.pt -save_model trained_models/democratic_generator -classifier_model ../models/classifier/political_classifier/political_classifier.pt -encoder_model ../models/translation/french_english/french_english.pt -tgt_label 1

# Translate the republican test set using the best democratic generator
python translate.py -encoder_model ../models/translation/french_english/french_english.pt -decoder_model ../models/style_generators/democratic_generator.pt -src ../data/political_data/republican_only.test.fr -output trained_models/republican_democratic.txt -replace_unk $true
