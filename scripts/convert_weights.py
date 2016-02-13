import cPickle as pickle

def convert_params(params):
    new_params = {}
    for param, value in params.items():
        new_params["%s" % param] = value.tolist()
    return new_params

if __name__ == "__main__":
    with open('old_models/charnet-top_2-1024-2.pkl', 'rb') as fp:
        generative_params = pickle.load(fp)
    lstm1 = convert_params(generative_params['lstm']['input_layer']['parameters'])
    lstm2 = convert_params(generative_params['lstm']['layers'][0]['parameters'])

    new_state = (lstm1, lstm2)
    with open('models/old-generative-model.pkl', 'wb') as fp:
        pickle.dump(new_state, fp)
