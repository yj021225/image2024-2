import cv2
from scipy.ndimage import label
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.python.data.util.nest import flatten

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

print(train_images.shape, train_labels.shape)
print(test_images.shape, test_labels.shape)

print(train_images[999])
print(train_labels[999])
cv2.imshow('cifar[3]', cv2.resize(train_images[999], (320,320)))
# cv2.waitKey(0)
cv2.destroyAllWindows()

val_images = train_images[45000:]
val_labels = train_labels[45000:]

# 처음모델, 총 파라매터 170만개
# mlp_model = Sequential([
#     Flatten(input_shape=(32, 32, 3)),
#     Dense(512, activation='relu'),
#     Dense(256, activation='relu'),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# 두번째모델 dropout 적용
# mlp_model = Sequential([
#     Flatten(input_shape=(32, 32, 3)),
#     Dense(512, activation='relu'),
#     Dropout(0.2),
#     Dense(256, activation='relu'),
#     Dropout(0.2),
#     Dense(128, activation='relu'),
#     Dropout(0.2),
#     Dense(10, activation='softmax')
# ])

# 단순 COnv2D 모델
# mlp_model = Sequential([
#     Conv2D(32, (3,3), padding='same',
#            activation='relu', input_shape=(32,32,3)),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# 동작하는 Conv2D 모델
# mlp_model = Sequential([
#     Conv2D(32, (3,3), padding='same',
#            activation='relu', input_shape=(32,32,3)),
#     MaxPooling2D((2,2)),
#     Conv2D(62, (3,3), padding='same', activation='relu'),
#     MaxPooling2D((2,2)),
#     Conv2D(64, (3,3), padding='same', activation='relu'),
#     Flatten(),
#     Dropout(0.3),
#     Dense(64, activation='relu'),
#     Dropout(0.5),
#     Dense(10, activation='softmax')
# ])

mlp_model = Sequential([

    Conv2D(128, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(2, 2),
    # The second convolution
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    # The third convolution
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dropout(0.5),
    # 512 neuron hidden layer
    Dense(500 , activation='relu'),

    # YOUR CODE HERE, BUT END WITH A 3 Neuron Dense, activated by softmax
    Dense(10, activation='softmax')
])

mlp_model.summary()
mlp_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# 텐서보드 실행준비
import datetime
import tensorflow as tf
log_dir= "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                      histogram_freq=1)
print(log_dir)
history = mlp_model.fit(train_images, train_labels, epochs=5,
                        validation_data=(val_images, val_labels),
                        callbacks=[tensorboard_callback])

print("Test결과")
mlp_model.evaluate(test_images, test_labels)
import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'], label = 'Train Accuracy')
# plt.plot(history.history['loss'], label = 'loss')
# plt.show()

predicted_labels = mlp_model.predict(test_images[:10])
print(predicted_labels)

predicted_results = tf.argmax(predicted_labels, axis=1)
print(predicted_results)