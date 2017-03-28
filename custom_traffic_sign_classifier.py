import cv2
import numpy as np
import matplotlib.image as mpimg
import traffic_sign_classifier_network as network

def load_and_pre_process(filename):
    loaded_img = cv2.cvtColor(cv2.resize(mpimg.imread(filename), (32, 32)), cv2.COLOR_RGB2GRAY)
    return np.expand_dims(np.array(loaded_img), 3)

container = []
test = load_and_pre_process('img1.jpg')
container.append(load_and_pre_process('img1.jpg'))
container.append(load_and_pre_process('img2.jpg'))
container.append(load_and_pre_process('img3.jpg'))
container.append(load_and_pre_process('img4.jpg'))
container.append(load_and_pre_process('img5.jpg'))

X_custom = np.array(container)
print("New Image Set Size: {}".format(X_custom.shape))
y_str = ['No entry', 'End of no passing', 'Stop', 'Speed limit (60km/h)', 'Yield']
y_custom = [17, 41, 14, 3, 13]
num_labels = len(y_custom)
print("Number of labels={}".format(num_labels))
print()
network.run_network(X_custom, y_custom, process_test=False, test_validation=True, print_test_logits=True)
network.top_3(X_custom)