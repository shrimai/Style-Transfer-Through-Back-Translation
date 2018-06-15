from __future__ import division

import argparse
import torch
import torch.nn as nn
from torch import cuda
from torch.autograd import Variable
import math
import time
import sys
import onmt

parser = argparse.ArgumentParser(description='train.py')

## Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-classifier_model', default='', type=str,
                    help="""If training from a classifier then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")
parser.add_argument('-encoder_model', required=True,
                    help='Path to the pretrained encoder model.')
parser.add_argument('-tgt_label', default=0, type=int,
                    help="""Specify the target label i.e the label of the 
                    decoder you are training for OR the label you want the
                    classifier to check.""")

## Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=300,
                    help='Word embedding sizes')
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder.""")
parser.add_argument('-brnn', action='store_true', default=True,
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-sequence_length', type=int, default=50,
                    help="""Max sequence kength for CNN. Give the one you gave
                            while constructing the CNN!""")

## Optimization options

parser.add_argument('-class_weight', type=float, default=1.0,
                    help='weight of the classifier loss')
parser.add_argument('-nll_weight', type=float, default=1.0,
                    help='weight of the cross entropy loss')
parser.add_argument('-temperature', type=float, default=1.0,
                    help='temperature for softmax')
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""Parameters are initialized over uniform distribution
                    with support (-param_init, param_init)""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-curriculum', action="store_true",
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")

#learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1, adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")

#pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")

opt = parser.parse_args()

print(opt)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, so you should probably run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])

def NLLLoss(vocabSize):
    weight = torch.ones(vocabSize)
    weight[onmt.Constants.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=True)
    if opt.gpus:
        crit.cuda()
    return crit

def BCELoss():
    crit = nn.BCELoss()
    if opt.gpus:
        crit.cuda()
    return crit

def memoryEfficientLoss(outputs, targets, model, crit1, crit2, eval=False):
    # compute generations one piece at a time
    num_correct, loss = 0, 0
    outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)
    softmax = nn.Softmax()
 
    generator_outputs = model.generator(outputs)
    linear = model.class_input(outputs)
    if len(opt.gpus) >= 1:
        softmax = softmax.cuda()
        curr_zeros = torch.cuda.FloatTensor(opt.sequence_length - linear.size(0),
                                           linear.size(1), linear.size(2)).zero_()
        # Create a batch_size long tensor filled with the label to be generated
        class_tgt = torch.cuda.FloatTensor(linear.size(1)).fill_(opt.tgt_label)
    else:
        curr_zeros = torch.FloatTensor(opt.sequence_length - linear.size(0), 
                                           linear.size(1), linear.size(2)).zero_()
        class_tgt = torch.FloatTensor(linear.size(1)).fill_(opt.tgt_label)

    curr_zeros = Variable(curr_zeros)

    linear_mod = linear.view(-1, linear.size(2))
    linear_mod = linear_mod.div(opt.temperature)
    soft_out = softmax(linear_mod)
    soft_out = soft_out.view(linear.size(0), linear.size(1), linear.size(2))
    #check if generated outputs are of max length
    if linear.size(0) < opt.sequence_length:
        soft_cat = torch.cat((soft_out, curr_zeros), 0)
    else:
        soft_cat = soft_out[:opt.sequence_length]
    class_outputs = model.class_model(soft_cat)
    batch_size = outputs.size(1)
    loss1 = crit1(generator_outputs.view(-1, generator_outputs.size(2)), targets.view(-1))
    class_tgt = Variable(class_tgt)
    loss2 = crit2(class_outputs, class_tgt)
    predicted_ids = generator_outputs.max(2)[1]
    num_correct = predicted_ids.data.eq(targets.data).masked_select(targets.ne(onmt.Constants.PAD).data).sum()
    loss = opt.nll_weight * loss1 + opt.class_weight * loss2

    if not eval:
        loss.backward()
    grad_output = None if outputs.grad is None else outputs.grad.data
    return loss1.data[0], loss2.data[0], grad_output, num_correct


def eval(model, crit1, crit2, data):
    total_loss = 0
    total_words = 0
    total_num_correct = 0

    model.eval()
    for i in range(len(data)):
        batch = data[i][:-1] # exclude original indices
        #  (1) run the encoder on the src
        encStates, context = model.encoder(batch[0])
        outputs = model(batch, encStates, context)
        targets = batch[1][1:]  # exclude <s> from targets
        loss, _, _, num_correct = memoryEfficientLoss(
                outputs, targets, model, crit1, crit2, eval=True)
        total_loss += loss
        total_num_correct += num_correct
        total_words += targets.data.ne(onmt.Constants.PAD).sum()

    model.train()
    return total_loss / len(data), total_num_correct / total_words


def trainModel(model, trainData, validData, dataset, optim):
    print(model)
    sys.stdout.flush()
    model.train()
    
    # define criterion of each GPU
    crit1 = NLLLoss(dataset['dicts']['tgt'].size())
    crit2 = BCELoss()

    start_time = time.time()
    def trainEpoch(epoch):
        if opt.extra_shuffle and epoch > opt.curriculum:
            trainData.shuffle()

        # shuffle mini batch order
        batchOrder = torch.randperm(len(trainData))       

        total_loss, total_words, total_num_correct = 0, 0, 0
        report_loss, report_closs, report_tgt_words, report_src_words, report_num_correct = 0, 0, 0, 0, 0
        start = time.time()
        for i in range(len(trainData)):

            batchIdx = batchOrder[i] if epoch > opt.curriculum else i
            batch = trainData[batchIdx][:-1] # exclude original indices

            model.zero_grad()
            #  (1) run the encoder on the src
            encStates, context = model.encoder(batch[0])
            outputs = model(batch, encStates, context)
            
            targets = batch[1][1:]  # exclude <s> from targets
            loss, closs, gradOutput, num_correct = memoryEfficientLoss(
                     outputs, targets, model, crit1, crit2)
            outputs.backward(gradOutput)

            
            # update the parameters
            optim.step()

            num_words = targets.data.ne(onmt.Constants.PAD).sum()
            report_loss += loss
            report_closs += closs
            report_num_correct += num_correct
            report_tgt_words += num_words
            report_src_words += sum(batch[0][1])
            total_loss += loss
            total_num_correct += num_correct
            total_words += num_words
            if i % opt.log_interval == -1 % opt.log_interval:
                print("Epoch %2d, %5d/%5d; acc: %6.2f; ppl: %6.2f; closs: %6.4f; %3.0f src tok/s; %3.0f tgt tok/s; %6.0f s elapsed" %
                      (epoch, i+1, len(trainData),
                      report_num_correct / report_tgt_words * 100,
                      math.exp(report_loss / opt.log_interval),
                      report_closs / opt.log_interval,
                      report_src_words/(time.time()-start),
                      report_tgt_words/(time.time()-start),
                      time.time()-start_time))
                
                sys.stdout.flush()
                report_loss = report_tgt_words = report_src_words = report_num_correct = report_closs = 0
                start = time.time()

        return total_loss / total_words, total_num_correct / total_words
    
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')
        #  (1) train for one epoch on the training set
        train_loss, train_acc = trainEpoch(epoch)
        train_ppl = math.exp(min(train_loss, 100))
        print('Train perplexity: %g' % train_ppl)
        print('Train accuracy: %g' % (train_acc*100))
        
        #  (2) evaluate on the validation set
        valid_loss, valid_acc = eval(model, crit1, crit2, validData)
        valid_ppl = math.exp(min(valid_loss, 100))
        print('Validation perplexity: %g' % valid_ppl)
        print('Validation accuracy: %g' % (valid_acc*100))
        
        sys.stdout.flush()
        #  (3) update the learning rate
        optim.updateLearningRate(valid_loss, epoch)

        model_state_dict = model.module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items() if 'generator' not in k}
        generator_state_dict = model.generator.module.state_dict() if len(opt.gpus) > 1 else model.generator.state_dict()
        #  (4) drop a checkpoint
        checkpoint = {
            'decoder': model.decoder.state_dict(),
            'generator': generator_state_dict,
            'dicts': dataset['dicts'],
            'opt': opt,
            'epoch': epoch,
            'optim': optim
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt' % (opt.save_model, 100*valid_acc, valid_ppl, epoch))




def main():

    print("Loading data from '%s'" % opt.data)
    
    dataset = torch.load(opt.data)

    dict_checkpoint = opt.train_from if opt.train_from else opt.train_from_state_dict
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus)
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True)

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Loading Encoder Model ...')
    enc_check = torch.load(opt.encoder_model, map_location=lambda storage, loc: storage)
    m_opt = enc_check['opt']
    src_dict = enc_check['dicts']['src']
    encoder = onmt.Models.Encoder(m_opt, src_dict)
    encoder.load_state_dict(enc_check['encoder'])

    print('Loading CNN Classifier Model ...')
    class_check = torch.load(opt.classifier_model, map_location=lambda storage, loc: storage)
    class_opt = class_check['opt']
    class_dict = class_check['dicts']['src']    
    class_model = onmt.CNNModels.ConvNet(class_opt, class_dict)
    class_model.load_state_dict(class_check['model'])

    print('Building model...')
    
    decoder = onmt.Models_decoder.Decoder(opt, dicts['tgt'])

    generator = nn.Sequential(
        nn.Linear(opt.rnn_size, dicts['tgt'].size()),
        nn.LogSoftmax())

    class_input = nn.Sequential(
        nn.Linear(opt.rnn_size, class_dict.size()))

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items() if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1

    if opt.train_from_state_dict:
        print('Loading model from checkpoint at %s' % opt.train_from_state_dict)
        decoder.load_state_dict(checkpoint['decoder'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1
    
    model = onmt.Models_decoder.DecoderModel(decoder)

    if len(opt.gpus) >= 1:
        encoder.cuda()
        model.cuda()
        class_model.cuda()
        generator.cuda()
        class_input.cuda()
    else:
        encoder.cpu()
        model.cpu()
        class_model.cpu()
        generator.cpu()
        class_input.cpu()

    if len(opt.gpus) > 1:
        encoder = nn.DataParallel(encoder, device_ids=opt.gpus, dim=1)
        model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
        generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)
        class_input = nn.DataParallel(class_input, device_ids=opt.gpus, dim=0)

    if not opt.train_from_state_dict and not opt.train_from:
        for p in model.parameters():
            p.data.uniform_(-opt.param_init, opt.param_init)

        decoder.load_pretrained_vectors(opt)

        optim = onmt.Optim(
            opt.optim, opt.learning_rate, opt.max_grad_norm,
            lr_decay=opt.learning_rate_decay,
            start_decay_at=opt.start_decay_at
        )
    else:
        print('Loading optimizer from checkpoint:')
        optim = checkpoint['optim']
        print(optim)

    optim.set_parameters(model.parameters())

    model.encoder = encoder
    model.generator = generator
    model.class_input = class_input
    model.class_model = class_model

    if opt.train_from or opt.train_from_state_dict:
        optim.optimizer.load_state_dict(checkpoint['optim'].optimizer.state_dict())

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)

    trainModel(model, trainData, validData, dataset, optim)

if __name__ == "__main__":
    main()
