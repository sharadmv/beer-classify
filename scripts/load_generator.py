from deepx.nn import *
from deepx.rnn import *
import cPickle as pickle

def convert_params(params):
    new_params = {}
    for param, value in params.items():
        new_params["%s" % param] = value.tolist()
    return new_params

def generate_sample(length):
    '''Generate a sample from the current version of the generator'''
    characters = [np.array([0])]
    model.reset_states()
    for i in xrange(length):
        output = model.predict(np.eye(len(text_encoding))[None, characters[-1]])
        sample = np.random.choice(xrange(len(text_encoding)), p=output[0, 0])
        characters.append(np.array([sample]))
    characters =  np.array(characters).ravel()
    return ''.join([text_encoding.decode(c) for c in characters[1:]])

def generate_sample2(length):
    '''Generate a sample from the current version of the generator'''
    return ''.join([text_encoding.decode(c) for c in model2.predict(np.eye(len(text_encoding))[None, 0]).argmax(axis=2).ravel()])

if __name__ == "__main__":
    with open('old_models/charnet-top_2-1024-2.pkl', 'rb') as fp:
        generative_params = pickle.load(fp)
    with open('old_models/charnet-encoding.pkl', 'rb') as fp:
        text_encoding = pickle.load(fp)

    lstm1 = convert_params(generative_params['lstm']['input_layer']['parameters'])
    lstm2 = convert_params(generative_params['lstm']['layers'][0]['parameters'])
    softmax = convert_params(generative_params['output']['parameters'])

    new_state = (({}, (lstm1, lstm2)), softmax)

    model = Sequence(Vector(len(text_encoding), 1)) >> Repeat(LSTM(1024, stateful=True), 2) >> Softmax(len(text_encoding))
    model.set_state(new_state)

    model2 = Generate(Vector(len(text_encoding)) >> Repeat(LSTM(1024), 2) >> Softmax(len(text_encoding)), 100)
    model2.set_state(new_state)
