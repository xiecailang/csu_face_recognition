import face_recognition
import pickle
#compare
pre_labels = []
count = 0
pre_lst = []

train_codings = []
train_lables = []
test_codings = []
test_labels = []

with open('../data/train_codings.pkl', 'rb') as f:
    train_codings = pickle.load(f)
with open('../data/train_labels.pkl', 'rb') as f:
    train_lables = pickle.load(f)
with open('../data/test_codings.pkl', 'rb') as f:
    test_codings = pickle.load(f)
with open('../data/test_labels.pkl', 'rb') as f:
    test_labels = pickle.load(f)

pre_labels = []
count = 0
for i in range(0, len(test_codings)):
    distances = face_recognition.face_distance(train_codings, test_codings[i])
    index = distances.__index__(min(distances))
    pre_labels.append(train_lables[index])
    if(train_lables[index] == test_labels[i]):
        count = count + 1

print('predict accuracy: ', count/len(test_labels))

