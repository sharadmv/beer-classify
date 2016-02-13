import cPickle as pickle
from path import Path
import logging

def load_data(data_dir, name='core'):

    logging.info("Loading all the data...")

    data_dir = Path(data_dir)
    with open(data_dir / 'beer_%s.pkl' % name, 'rb') as fp:
        all_reviews = pickle.load(fp)[0]

    all_reviews = [a for a in all_reviews if a.text != '']

    logging.info("Loading the training data...")

    with open(data_dir / 'beer_%s-train.pkl' % name, 'rb') as fp:
        train_reviews = pickle.load(fp)[0]

    train_reviews = [a for a in train_reviews if a.text != '']

    logging.info("Loading the test data...")

    with open(data_dir / 'beer_%s-test.pkl' % name, 'rb') as fp:
        test_reviews = pickle.load(fp)[0]

    test_reviews = [a for a in test_reviews if a.text != '']

    return all_reviews, train_reviews, test_reviews
