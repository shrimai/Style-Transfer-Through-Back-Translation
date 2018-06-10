from __future__ import division

import onmt
import torch
import argparse
import math
import codecs
import sys

parser = argparse.ArgumentParser(description='translate.py')

parser.add_argument('-model', required=True,
                    help='Path to model .pt file')
parser.add_argument('-num_classes', default=2, type=int,
                    help="""Number of classes""")
parser.add_argument('-src',   required=True,
                    help='Source sequence to check')
parser.add_argument('-tgt',   required=True,
                    help='The target label for the classifier check')
parser.add_argument('-label0',   required=True,
                    help='Label for 0')
parser.add_argument('-label1',   required=True,
                    help='Label for 1')
parser.add_argument('-output', default=None,
                    help="""Path to output the predictions (each line will
                    be the decoded sequence""")
parser.add_argument('-batch_size', type=int, default=30,
                    help='Batch size')
parser.add_argument('-max_sent_length', type=int, default=50,
                    help='Maximum sentence length.')
parser.add_argument('-verbose', action="store_true",
                    help='Print scores and predictions for each sentence')
parser.add_argument('-gpu', type=int, default=-1,
                    help="Device to run on")



def reportScore(name, scoreTotal, wordsTotal):
    print("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
        name, scoreTotal / wordsTotal,
        name, math.exp(-scoreTotal/wordsTotal)))

def addone(f):
    for line in f:
        yield line
    yield None

def main():
    opt = parser.parse_args()
    opt.cuda = opt.gpu > -1
    if opt.cuda:
        torch.cuda.set_device(opt.gpu)

    translator = onmt.Translator_cnn(opt)

    srcBatch, tgtBatch = [], []

    count = 0
    total_correct, total_words, total_loss = 0, 0, 0
    outputs, predictions, sents = [], [], []
    for line in addone(codecs.open(opt.src, "r", "utf-8")):
        count += 1
        if line is not None:
            sents.append(line)
            srcTokens = line.split()
            if len(srcTokens) <= opt.max_sent_length:
                srcBatch += [srcTokens]
            else:
                srcBatch += [srcTokens[:opt.max_sent_length]]
            tgtBatch += [opt.tgt]

            if len(srcBatch) < opt.batch_size:
                continue
        else:
            # at the end of file, check last batch
            if len(srcBatch) == 0:
                break
        num_correct, batchSize, outs, preds = translator.translate(srcBatch, tgtBatch)
 
        total_correct += num_correct
        total_words += batchSize
        outputs += outs.data.tolist()
        predictions += preds.tolist()
    

        srcBatch, tgtBatch = [], []
        if count%1000 == 0:
            print('Completed: ', str(count))
            sys.stdout.flush()
    if opt.output:
        with open(opt.output, "w") as outF:
            for i in range(len(sents)):
                outF.write(str(predictions[i]) + "\t" + str(outputs[i]) + "\t" + sents[i])

    print('Accuracy: ', str((total_correct*100)/total_words))
    print('')


if __name__ == "__main__":
    main()
