import tensorflow as tf
mnist = tf.keras.datasets.mnist #download minist


(x_train, y_train),(x_test,y_test)=mnist.load_data()# split data 

x_train = tf.keras.utils.normalize(x_train , axis = 1)
x_test = tf.keras.utils.normalize(x_test , axis = 1)

x_train = x_train / 255
x_test = x_test / 255

#cnn model
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=25, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D((2, 2)))

model.add(tf.keras.layers.Dense(units=64,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

model.compile(optimizer='adam' , loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train, epochs=3)
model.evaluate(x_test, y_test)
#predict
y_predicted_by_model = model.predict(x_test)
y_predicted_by_model[0]




