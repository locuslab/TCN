import argparse
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
sys.path.append("../../")
from TCN.lambada_language.utils import *
from TCN.lambada_language.model import TCN
import pickle


parser = argparse.ArgumentParser(description='Sequence Modeling - LAMBADA Textual Understanding')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size (default: 20)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to the embedded layer (default: 0.1)')
parser.add_argument('--clip', type=float, default=0.4,
                    help='gradient clip, -1 means no clip (default: 0.4)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=4,
                    help='kernel size (default: 4)')
parser.add_argument('--data', type=str, default='./data/lambada',
                    help='location of the data corpus (default: ./data/lambada)')
parser.add_argument('--emsize', type=int, default=500,
                    help='size of word embeddings (default: 500)')
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 5)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100)')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--nhid', type=int, default=500,
                    help='number of hidden units per layer (default: 500)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights (default: True)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer type (default: SGD)')
parser.add_argument('--validseqlen', type=int, default=50,
                    help='valid sequence length (default: 50)')
parser.add_argument('--seq_len', type=int, default=100,
                    help='total sequence length, including effective history (default: 100)')
parser.add_argument('--corpus', action='store_true',
                    help='force re-make the corpus (default: False)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


print(args)
train_data, val_data, test_data, corpus = data_generator(args)

n_words = len(corpus.dictionary)
print("Total # of words: {0}".format(n_words))

num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
tied = args.tied

model = TCN(args.emsize, n_words, num_chans, dropout=dropout, 
            emb_dropout=emb_dropout, kernel_size=k_size, tied_weights=tied)

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(data_source):
    model.eval()
    total_loss = 0
    processed_data_size = 0
    correct = 0
    with torch.no_grad():
        for i in range(len(data_source)):
            data, targets = torch.LongTensor(data_source[i]).view(1, -1), torch.LongTensor([data_source[i][-1]]).view(1, -1)
            data, targets = Variable(data), Variable(targets)
            if args.cuda:
                data, targets = data.cuda(), targets.cuda()
            output = model(data)
            final_output = output[:, -1].contiguous().view(-1, n_words)
            final_target = targets[:, -1].contiguous().view(-1)
            loss = criterion(final_output, final_target)
            total_loss += loss.data
            processed_data_size += 1
        return total_loss.item() / processed_data_size


def train():
    global train_data
    model.train()
    total_loss = 0
    start_time = time.time()
    for babatch_idxtch, i in enumerate(range(0, train_data.size(1) - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= train_data.size(1) - 1:
            continue
        data, targets = get_batch(train_data, i, args)
        optimizer.zero_grad()
        output = model(data)
        eff_history = args.seq_len - args.validseqlen
        if eff_history < 0:
            raise ValueError("Valid sequence length must be smaller than sequence length!")
        final_target = targets[:, eff_history:].contiguous().view(-1)
        final_output = output[:, eff_history:].contiguous().view(-1, n_words)
        loss = criterion(final_output, final_target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.5f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch_idx, train_data.size(1) // args.validseqlen, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            reg_loss = 0
            start_time = time.time()


if __name__ == "__main__":
    best_vloss = 1e8
    try:
        all_vloss = []
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_loss = evaluate(val_data)
            test_loss = evaluate(test_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('| end of epoch {:3d} | time: {:5.2f}s | test loss {:5.2f} | '
                  'test ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                            test_loss, math.exp(test_loss)))
            print('-' * 89)
            # Save the model if the validation loss is the best we've seen so far.

            if val_loss < best_vloss:
                with open("model.pt", 'wb') as f:
                    print('Save model!\n')
                    torch.save(model, f)
                best_vloss = val_loss
            if epoch > 5 and val_loss >= max(all_vloss[-5:]):
                lr = lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_vloss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open("model.pt", 'rb') as f:
        model = torch.load(f)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
