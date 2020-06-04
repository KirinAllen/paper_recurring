from recurrent_of_paper import read_data_sets
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras_preprocessing.image import ImageDataGenerator

#read the dataset
data = read_data_sets('MNIST_data', n_labeled=900, one_hot=True)

train_datagen = ImageDataGenerator(
    rotation_range = 40,
    width_shift_range= 0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    data.train.S0.images.reshape(300,28,28,1),
    y=data.train.S0.labels
)

validation_generator = validation_datagen.flow(
    data.validation.images.reshape(5000,28,28,1),
    y=data.validation.labels
)

#define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(16,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation='softmax')

])

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
history = model.fit_generator(train_generator,epochs=15,validation_data=validation_generator,verbose=1)
