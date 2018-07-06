import csv
import face_recognition
import pickle

"""
csv_reader = csv.reader(open('../data/train.csv', encoding='utf-8'))
train_imgs = []
train_img_codings = []
train_labels = []
count = 0
try:
    for item in csv_reader:
        img_path = '%s%s' % ('../data/train/', item[0])
        img = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(img, model="cnn")
        if len(face_locations) > 0 :
            img_code = face_recognition.face_encodings(img,face_locations)[0]
            train_imgs.append(img)
            train_img_codings.append(img_code)
            train_labels.append(item[1])
            print('training', item[1])
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")

# 序列化
with open('../data/train_codings.pkl', 'wb') as f:
    pickle.dump(train_img_codings, f)
with open('../data/train_labels.pkl', 'wb') as f:
    pickle.dump(train_labels, f)
"""
#load test files
test_gallery_imgs = []
test_img_codings = []
test_gallery_labels = []
csv_reader = csv.reader(open('../data/test_a_gallery.csv', encoding='utf-8'))

try:
    for item in csv_reader:
        img_path = '%s%s' % ('../data/test_a/gallery/', item[0])
        img = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(img, model="cnn")
        if len(face_locations) > 0:
            test_gallery_imgs.append(img)
            test_img_codings.append(face_recognition.face_encodings(img, face_locations)[0])
            test_gallery_labels.append(item[1])
            print('testing', item[1])
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")

# 序列化
with open('../data/test_codings.pkl', 'wb') as f:
    pickle.dump(test_img_codings, f)
with open('../data/test_labels.pkl', 'wb') as f:
    pickle.dump(test_gallery_labels, f)

