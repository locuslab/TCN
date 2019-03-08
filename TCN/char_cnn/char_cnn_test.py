import argparse
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append("../../")
from TCN.char_cnn.utils import *
from TCN.char_cnn.model import TCN
import time
import math


import warnings
warnings.filterwarnings("ignore")   # Suppress the RunTimeWarning on unicode


parser = argparse.ArgumentParser(description='Sequence Modeling - Character Level Language Model')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--emb_dropout', type=float, default=0.1,
                    help='dropout applied to the embedded layer (0 = no dropout) (default: 0.1)')
parser.add_argument('--clip', type=float, default=0.15,
                    help='gradient clip, -1 means no clip (default: 0.15)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=3,
                    help='kernel size (default: 3)')
parser.add_argument('--levels', type=int, default=3,
                    help='# of levels (default: 3)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=4,
                    help='initial learning rate (default: 4)')
parser.add_argument('--emsize', type=int, default=100,
                    help='dimension of character embeddings (default: 100)')
parser.add_argument('--optim', type=str, default='SGD',
                    help='optimizer to use (default: SGD)')
parser.add_argument('--nhid', type=int, default=450,
                    help='number of hidden units per layer (default: 450)')
parser.add_argument('--validseqlen', type=int, default=320,
                    help='valid sequence length (default: 320)')
parser.add_argument('--seq_len', type=int, default=400,
                    help='total sequence length, including effective history (default: 400)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--dataset', type=str, default='ptb',
                    help='dataset to use (default: ptb)')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")


print(args)
file, file_len, valfile, valfile_len, testfile, testfile_len, corpus = data_generator(args)

n_characters = len(corpus.dict)
train_data = batchify(char_tensor(corpus, file), args.batch_size, args)
val_data = batchify(char_tensor(corpus, valfile), 1, args)
test_data = batchify(char_tensor(corpus, testfile), 1, args)
print("Corpus size: ", n_characters)


num_chans = [args.nhid] * (args.levels - 1) + [args.emsize]
k_size = args.ksize
dropout = args.dropout
emb_dropout = args.emb_dropout
model = TCN(args.emsize, n_characters, num_chans, kernel_size=k_size, dropout=dropout, emb_dropout=emb_dropout)


if args.cuda:
    model.cuda()


criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def evaluate(source):
    model.eval()
    total_loss = 0
    count = 0
    source_len = source.size(1)
    with torch.no_grad():
        for batch, i in enumerate(range(0, source_len - 1, args.validseqlen)):
            if i + args.seq_len - args.validseqlen >= source_len:
                continue
            inp, target = get_batch(source, i, args)
            output = model(inp)
            eff_history = args.seq_len - args.validseqlen
            final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
            final_target = target[:, eff_history:].contiguous().view(-1)
            loss = criterion(final_output, final_target)

            total_loss += loss.data * final_output.size(0)
            count += final_output.size(0)

        val_loss = total_loss.item() / count * 1.0
        return val_loss


def train(epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    losses = []
    source = train_data
    source_len = source.size(1)
    for batch_idx, i in enumerate(range(0, source_len - 1, args.validseqlen)):
        if i + args.seq_len - args.validseqlen >= source_len:
            continue
        inp, target = get_batch(source, i, args)
        optimizer.zero_grad()
        output = model(inp)
        eff_history = args.seq_len - args.validseqlen
        final_output = output[:, eff_history:].contiguous().view(-1, n_characters)
        final_target = target[:, eff_history:].contiguous().view(-1)
        loss = criterion(final_output, final_target)
        loss.backward()

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            losses.append(cur_loss)
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.3f} | bpc {:5.3f}'.format(
                epoch, batch_idx, int((source_len-0.5) / args.validseqlen), lr,
                              elapsed * 1000 / args.log_interval, cur_loss, cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()

    return sum(losses) * 1.0 / len(losses)


def main():
    global lr
    try:
        print("Training for %d epochs..." % args.epochs)
        all_losses = []
        best_vloss = 1e7
        for epoch in range(1, args.epochs + 1):
            loss = train(epoch)

            vloss = evaluate(val_data)
            print('-' * 89)
            print('| End of epoch {:3d} | valid loss {:5.3f} | valid bpc {:8.3f}'.format(
                epoch, vloss, vloss / math.log(2)))

            test_loss = evaluate(test_data)
            print('=' * 89)
            print('| End of epoch {:3d} | test loss {:5.3f} | test bpc {:8.3f}'.format(
                epoch, test_loss, test_loss / math.log(2)))
            print('=' * 89)

            if epoch > 5 and vloss > max(all_losses[-3:]):
                lr = lr / 10.
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            all_losses.append(vloss)

            if vloss < best_vloss:
                print("Saving...")
                save(model)
                best_vloss = vloss

    except KeyboardInterrupt:
        print('-' * 89)
        print("Saving before quit...")
        save(model)

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.3f} | test bpc {:8.3f}'.format(
        test_loss, test_loss / math.log(2)))
    print('=' * 89)

# train_by_random_chunk()
if __name__ == "__main__":
    main()
