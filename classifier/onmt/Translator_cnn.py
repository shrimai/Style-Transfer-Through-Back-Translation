import onmt
import torch.nn as nn
import torch
from torch.autograd import Variable


class Translator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch

        self.label0 = opt.label0
        self.label1 = opt.label1

        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['opt']
        self.src_dict = checkpoint['dicts']['src']

        model = onmt.CNNModels.ConvNet(model_opt, self.src_dict)
        model.load_state_dict(checkpoint['model'])

        if opt.cuda:
            model.cuda()
            self.gpu = True
        else:
            model.cpu()
            self.gpu = False

        self.model = model
        self.model.eval()

    def buildData(self, srcBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                    onmt.Constants.UNK_WORD, padding=True) for b in srcBatch]
        tgtData = []
        if goldBatch:
            for label in goldBatch:
                if label == self.label0:
                    tgtData += [torch.LongTensor([0])]
                elif label == self.label1:
                    tgtData += [torch.LongTensor([1])]

        return onmt.Dataset(srcData, tgtData,
            self.opt.batch_size, self.opt.cuda, volatile=True)

    def translateBatch(self, srcBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        Batch = (srcBatch, tgtBatch)

        inp = srcBatch[0] % self.src_dict.size()
        inp_ = torch.unsqueeze(inp, 2)
        if self.gpu:
           one_hot = Variable(torch.cuda.FloatTensor(srcBatch[0].size(0), srcBatch[0].size(1), self.src_dict.size()).zero_())
        else:
           one_hot = Variable(torch.FloatTensor(srcBatch[0].size(0), srcBatch[0].size(1), self.src_dict.size()).zero_())
        one_hot_scatt = one_hot.scatter_(2, inp_, 1)

        outputs= self.model(one_hot_scatt)
        targets = tgtBatch
        outputs = Variable(outputs.data, requires_grad=False, volatile=False)
        if self.gpu:
            pred_t = torch.ge(outputs.data, torch.cuda.FloatTensor(outputs.size(0), outputs.size(1)).fill_(0.5))
        else:
            pred_t = torch.ge(outputs.data, torch.FloatTensor(outputs.size(0), outputs.size(1)).fill_(0.5))
        num_correct = pred_t.long().squeeze(-1).eq(targets[0].data).sum()
        return num_correct, batchSize, outputs, pred_t

    def translate(self, srcBatch, goldBatch, flag=False):
        #  (1) convert words to indexes
        dataset = self.buildData(srcBatch, goldBatch)
        src, tgt, indices = dataset[0]

        #  (2) translate
        num_correct, batchSize, outs, pred = self.translateBatch(src, tgt)

        return num_correct, batchSize, outs, pred
