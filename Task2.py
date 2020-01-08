from sklearn import  datasets,svm,metrics
import matplotlib.pyplot as plt
digits = datasets.load_digits()

#Task - a 认识数据集
images_and_labels = list(zip(digits.images, digits.target))
# for every element in the list
for index, (image, label) in enumerate(images_and_labels[:8]):
    # initialize a subplot of 2X4 at the i+1-th position
    plt.subplot(2, 4, index + 1)
    # Don't plot any axes
    plt.axis('off')
    # Display images in all subplots
    plt.imshow(image, cmap=plt.cm.gray_r,interpolation='nearest')
    # Add a title to each subplot
    plt.title('Training: ' + str(label))
plt.show()

#Task - b 注释已给程序
# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images) #获取数据维度
data = digits.images.reshape((n_samples, -1)) #将数据维度改成（样本数*64）

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001) #gamma是1/（2*σ^2） gamma越大支持向量越少，这个值会影响训练速度

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2]) #训练SVM

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]   #将数据对半分用作训练和预测
predicted = classifier.predict(data[n_samples // 2:])

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted)) #对角线上是预测正确数量 其他的是判断错成其他数字数量

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted)) #观察预测分类对应关系
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()

#Task - c 用KNN对手写数据集分类并识别，讨论k变化时分类性能变化

#Task - d 用SVM分类，对比最佳KNN性能好坏