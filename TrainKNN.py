from module.PuenBan_K_Tua import trainKNN, StatModel, SVM
import numpy as np
import os
import cv2

trainKNN = trainKNN()

dirSep = os.path.sep
path = os.getcwd() + dirSep + 'savedModel' + dirSep + 'modelKNN'

print('Loading digits from digits.png ... ')
# Load data.
digits, labels = trainKNN.load_digits( path + dirSep  + 'digits.png')
print(labels.shape)

print('Shuffle data ... ')
# Shuffle data
rand = np.random.RandomState(10)
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

print('Deskew images ... ')
digits_deskewed = list(map(trainKNN.deskew, digits))

print('Defining HoG parameters ...')
# HoG feature descriptor
hog = trainKNN.get_hog()

print('Calculating HoG descriptor for every image ... ')
hog_descriptors = []
for img in digits_deskewed:
    hog_descriptors.append(hog.compute(img))
hog_descriptors = np.squeeze(hog_descriptors)
print(hog_descriptors.shape)

print('Spliting data into training (90%) and test set (10%)... ')
train_n=int(0.9*len(hog_descriptors))
digits_train, digits_test = np.split(digits_deskewed, [train_n])
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [train_n])
labels_train, labels_test = np.split(labels, [train_n])


print('Training SVM model ...')
model = SVM()

print(hog_descriptors_train)
print(hog_descriptors_test.shape)
model.train(hog_descriptors_train, labels_train)

print('Saving SVM model ...')
model.save(path + dirSep + 'digits_svm.dat')


print('Evaluating model ... ')
vis = trainKNN.evaluate_model(model, digits_test, hog_descriptors_test, labels_test)
cv2.imwrite(path + dirSep +"digits-classification.jpg",vis)
cv2.imshow("Vis", vis)
cv2.waitKey(0)
