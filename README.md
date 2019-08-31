# Style Transfer Through Back-Translation

This repo contains the code and data of the following paper:
>Style Transfer Through Back-Translation. *Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan Salakhutdinov, Alan W Black*. ACL 2018. [arXiv](https://arxiv.org/pdf/1804.09000.pdf)

## Dependencies

- Python 3.6
- Pytorch 0.3

## Trained Machine Translation Models

- Dowload the english--french and french--english models from the following link:
```bash
http://tts.speech.cs.cmu.edu/style_models/english_french.tar
http://tts.speech.cs.cmu.edu/style_models/french_english.tar
```
Place these models in the `models/translation` folder.

## Trained Classifier Models

- Dowload the trained gender, political slant and sentiment classifiers from the following link:
```bash
http://tts.speech.cs.cmu.edu/style_models/gender_classifier.tar
http://tts.speech.cs.cmu.edu/style_models/political_classifier.tar
http://tts.speech.cs.cmu.edu/style_models/sentiment_classifier.tar
```
Place these models in the `models/classifier`folder.
The three classifiers are trained for the following labels:

 Task | Label = 0 | Label = 1 
 --- | --- | --- 
 Gender | Male | Female 
 Political | Republican | Democratic 
 Sentiment | Negative | Positive 

## Trained Style Models

- Download the trained style models from the following links:
```bash
http://tts.speech.cs.cmu.edu/style_models/female_generator.tar
http://tts.speech.cs.cmu.edu/style_models/male_generator.tar
http://tts.speech.cs.cmu.edu/style_models/democratic_generator.tar
http://tts.speech.cs.cmu.edu/style_models/republican_generator.tar
http://tts.speech.cs.cmu.edu/style_models/positive_generator.tar
http://tts.speech.cs.cmu.edu/style_models/negative_generator.tar
```
Place these models in the `models/style_generators` folder.

## Quick Start

Refer to example.sh file to see the commands.
- First `cd style_decoder` and then preprocess your raw data using the following command:   
```bash
python preprocess.py -train_src TRAIN_SOURCE_FILE -train_tgt TRAIN_TARGET_FILE -valid_src VALID_SOURCE_FILE -valid_tgt VALID_TARGET_FILE -save_data DATA_NAME
```
- Then train your model using the following command:
```bash
python train_decoder.py -data DATA_NAME.train.pt -save_model MODEL_DIR/MODEL_NAME -classifier_model CLASSIFIER.pt -encoder_model ENCODER_MODEL -tgt_label {0/1}
```

## Data

Dowload the data required for the political slant transfer experiment from the following link and place it in data/ folder. 
```bash
http://tts.speech.cs.cmu.edu/style_models/political_data.tar
tar -xvf political_data.tar
```
The train, dev, test and classtrain splits are given as is. If you are using this data then please cite the following papers:

    @inproceedings{style_transfer_acl18,
      title={Style Transfer Through Back-Translation},
      author={Prabhumoye, Shrimai and Tsvetkov, Yulia and Salakhutdinov, Ruslan and Black, Alan W},
      year={2018},
      booktitle={Proc. ACL}
      }

    @inproceedings{rtgender,
      title={{RtGender}: A Corpus for Studying Differential Responses to Gender},
      author={Voigt, Rob and Jurgens, David and Prabhakaran, Vinodkumar and Jurafsky, Dan and Tsvetkov, Yulia},
      year={2018},
      booktitle={Proc. LREC},
      }

Dowload the data required for the gender transfer experiment from the following link and place it in data/ folder.
```bash
http://tts.speech.cs.cmu.edu/style_models/gender_data.tar
tar -xvf gender_data.tar
```
The train, dev, test and classtrain splits are given as is. If you are using this data then please cite the following papers:

    @inproceedings{style_transfer_acl18,
      title={Style Transfer Through Back-Translation},
      author={Prabhumoye, Shrimai and Tsvetkov, Yulia and Salakhutdinov, Ruslan and Black, Alan W},
      year={2018},
      booktitle={Proc. ACL}
      }
      
    @inproceedings{reddy2016obfuscating,
      title={Obfuscating gender in social media writing},
      author={Reddy, Sravana and Knight, Kevin},
      year={2016},
      booktitle={Proc. of Workshop on Natural Language Processing and Computational Social Science}
      pages={17--26},
      }
      
You can find the data used in the sentiment modification experiment described in the paper at this [link](https://github.com/shentianxiao/language-style-transfer/tree/master/data/yelp). The train, dev, test and classtrain splits are given as is.

Download the data used in [Multiple-Attribute Text Rewriting](https://openreview.net/pdf?id=H1g2NhC5KQ) paper from the following link.
```
http://tts.speech.cs.cmu.edu/style_models/yelp_reviews.txt
http://tts.speech.cs.cmu.edu/style_models/yelp_attrs.txt
```

## Acknowledgements

The code used to train the NMT systems is from the [OpenNMT](https://github.com/OpenNMT/OpenNMT-py)  toolkit. This code base is based on the code of the toolkit.
