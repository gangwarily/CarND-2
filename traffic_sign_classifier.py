import pickle
import traffic_sign_classifier_network as network

testing_file = 'test.p'
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    X_test, y_test = test['features'], test['labels']

network.run_network(X_test, y_test, test_validation=True)
