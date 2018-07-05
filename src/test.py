import csv
import face_recognition

#load trainning files
csv_reader = csv.reader(open('../data/train.csv', encoding='utf-8'))
train_imgs = []
train_img_codings = []
train_labels = []
for item in csv_reader:
    img_path = '%s%s' % ('../data/train/', item[0])
    img = face_recognition.load_image_file(img_path)
    img_code = face_recognition.face_encodings(img)[0]

    train_imgs.append(img)
    train_img_codings.append(img_code)
    train_labels = item[1]

#load test files
test_gallery_imgs = []
test_img_codings = []
test_gallery_labels = []
csv_reader = csv.reader(open('../data/test_a_gallery.csv', encoding='utf-8'));
for item in csv_reader:
    img_path = '%s%s' % ('../data/test_a/gallery/',item[0])
    img = face_recognition.load_image_file(img_path)
    test_gallery_imgs.append(img)
    test_img_codings.append(face_recognition.face_encodings(img)[0])
    test_gallery_labels.append(item[1])

#compare
pre_labels = []
count =0;
for i in range(0,len(test_gallery_imgs)):
    t_img = test_gallery_imgs[i]
    pre_res = face_recognition.compare_faces(train_imgs,t_img)
    for j in range(0,len(pre_res)):
        if pre_res[j] == True:
            pre_labels.append(train_labels[j])
            if(train_labels[j] == test_gallery_labels[i]):
                count = count + 1

print(count,count/len(test_gallery_imgs))






