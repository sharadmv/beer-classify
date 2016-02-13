import cPickle as pickle
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")
sns.despine()
import logging
logging.basicConfig(level=logging.INFO)
from argparse import ArgumentParser
from path import Path
from dataset.beer import *
from dataset.encoding import *
from dataset.sequence import *
from dataset.batch import *
from tqdm import tqdm

from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

from load_data import load_data

def parse_args():
    argparser = ArgumentParser()

    argparser.add_argument('--model', default='itemnet')
    argparser.add_argument('--data_dir', default='data/')
    argparser.add_argument('--out_dir', default='out/')
    argparser.add_argument('--gain', action='store_true')
    argparser.add_argument('--load', default=None)
    argparser.add_argument('--validate', action='store_true')
    argparser.add_argument('--validation_split', default=0.1)
    argparser.add_argument('--iterations', default=100, type=int)

    return argparser.parse_args()

def noop(i):
    pass


def chunks(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def validate(model, validation_reviews, review_encoding, validation_targets, target_encoding):
    correct = []
    predictions = []
    for review, target in tqdm(zip(validation_reviews, validation_targets)[:100]):
        review = np.vstack([review_encoding.convert_representation(i) for i in review.seq])[:, np.newaxis]
        target = target.seq[0]
        pred = model.predict(review)[-1]
        print pred.shape
        result = target_encoding.decode(pred.argmax())
        print "Target:", target
        print "Result:", result
        print
        predictions.append(result)
        correct.append(result == target)
        # model.reset_states()
    return correct, predictions

class Trainer(object):

    def __init__(self, optimizer, model, batcher, learning_curve=None, callback=noop):
        self.optimizer = optimizer
        self.model = model
        self.batcher = batcher
        self.learning_curve = learning_curve
        self.callback = callback
        self.losses = []
        self.running_losses = []
        self.total_iterations = 0
        self.running_loss = None
        self.current_loss = None

    def train(self, n_iter, *args):
        loss = []
        for i in xrange(n_iter):
            X, y = self.batcher.next_batch()
            self.current_loss = self.optimizer.train(X, y, *args)
            self.losses.append(self.current_loss)
            loss.append(self.current_loss)
            if self.running_loss is None:
                self.running_loss = self.current_loss
            else:
                self.running_loss = 0.95 * self.running_loss + 0.05 * self.current_loss
            self.running_losses.append(self.running_loss)
            logging.info("Iteration %u: %f [%f]" % (
                i,
                self.current_loss,
                self.running_loss
            ))
            self.total_iterations += 1
            if self.total_iterations % 10 == 0:
                self.plot_learning_curve()
            if self.total_iterations % 100 == 0:
                with open(Path('out') / 'models' / 'train.pkl', 'wb') as fp:
                    pickle.dump(self.model.get_state(), fp)
        return loss

    def train_epoch(self, n_epochs, *args):
        loss = []
        for i in xrange(n_epochs):
            loss.extend(self.train(self.batcher.num_batches, *args))
            self.callback(i)
        self.plot_learning_curve()
        self.model.reset_states()
        return loss

    def plot_learning_curve(self):
        plt.figure()
        plt.plot(self.running_losses)
        plt.xlabel("Iterations")
        plt.ylabel("Average Cross-Entropy Loss")
        plt.savefig(self.learning_curve, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    args = parse_args()

    all_reviews, train_reviews, test_reviews = load_data(args.data_dir)

    out_dir = Path(args.out_dir)

    split = int((1 - args.validation_split) * len(train_reviews))
    train_reviews, valid_reviews = train_reviews[:split], train_reviews[split:]

    def get_sequences(reviews):
        if args.model == 'itemnet':
            def get_target(review):
                return SingletonSequence(review.beer.name)
        elif args.model == 'usernet':
            def get_target(review):
                return SingletonSequence(review.user)
        elif args.model == 'rating':
            def get_target(review):
                return NumberSequence([review.rating_overall])
        elif args.model == 'category':
            def get_target(review):
                return SingletonSequence(review.beer.style)
        return [CharacterSequence(r.text.replace('\x05', '').replace('\x0c', '')) for r in reviews], [get_target(r) for r in reviews]

    def encode_sequences(sequences, encoding):
        return [s.encode(encoding) for s in sequences]


    all_review_seqs, all_review_targets = get_sequences(all_reviews)
    # if args.model == 'rating':
        # train_reviews = [t for t in train_reviews if (t.rating_overall >= 4 or t.rating_overall <= 2)]
    train_review_seqs, train_review_targets = get_sequences(train_reviews)
    valid_review_seqs, valid_review_targets = get_sequences(valid_reviews)
    test_review_seqs, test_review_targets  = get_sequences(test_reviews)

    with open('old_models/charnet-encoding.pkl', 'rb') as fp:
        text_encoding = pickle.load(fp)

    if args.model == 'rating':
        target_encoding = IdentityEncoding(1)
        # target_encoding.build_encoding(all_review_targets)
    else:
        target_encoding = OneHotEncoding(include_start_token=False,
                                    include_stop_token=False)
        target_encoding.build_encoding(all_review_targets)
        train_review_targets = encode_sequences(train_review_targets, target_encoding)

    train_num_seqs = encode_sequences(train_review_seqs, text_encoding)
    valid_num_seqs = encode_sequences(valid_review_seqs, text_encoding)
    test_num_seqs = encode_sequences(test_review_seqs, text_encoding)


    train_review_targets = NumberSequence(
        np.concatenate([c.replicate(len(r)).seq for c, r in zip(train_review_targets, train_num_seqs)])
    )
    lengths = np.concatenate([np.linspace(0, 1, len(t)) for t in train_num_seqs])
    train_num_seqs = NumberSequence(np.concatenate([t.seq for t in train_num_seqs]))

    BATCH_SIZE = 256
    if args.validate:
        BATCH_SIZE = 1

    batcher = WindowedBatcher(train_num_seqs, text_encoding, train_review_targets, target_encoding,
                              lengths=lengths, batch_size=BATCH_SIZE,
                              sequence_length=200)

    logging.info("Compiling model...")

    lstm = Sequence(Vector(len(text_encoding), batch_size=BATCH_SIZE)) >> Repeat(LSTM(1024, stateful=True), 2) >> Softmax(len(target_encoding))
    if args.load is not None:
        with open(args.load, 'rb') as fp:
            lstm.set_state(pickle.load(fp))
    else:
        with open('models/old-generative-model.pkl', 'rb') as fp:
            lstm.left.right.set_state(pickle.load(fp))

    if args.validate:
        correct, results = validate(lstm, valid_num_seqs, text_encoding, valid_review_targets, target_encoding)
    else:
        rmsprop = RMSProp(lstm, CrossEntropy(), clip_gradients=3)

        def train_cb(i):
            logging.info("Dumping model %u" % i)
            with open(out_dir / 'models' / '%s-%u.pkl' % (args.model, i), 'wb') as fp:
                pickle.dump(lstm.get_state(), fp)

        trainer = Trainer(rmsprop, lstm, batcher, learning_curve=out_dir / ("%s-learning.png" % args.model), callback=train_cb)
        # trainer.train_epoch(4, 10)
