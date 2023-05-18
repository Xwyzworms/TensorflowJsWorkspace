import * as tf from '@tensorflow/tfjs';


const LinearRegression = tf.sequential();
LinearRegression.add(tf.layers.dense( {
    units : 1,
    inputShape : [1]
}))

const tensorX = tf.tensor([4, 1, 2, 3])
const tensorBias = tf.tensor([10])
const tensorY = tf.add(tensorX, tensorBias)

const lr = 0.8
const EPOCHS = 10
LinearRegression.compile({optimizer : tf.train.sgd(lr),
                          loss : 'meanSquaredError'

}) 

LinearRegression.fit(tensorX, tensorY, {
    epochs : EPOCHS
})

LinearRegression.predict(tf.tensor([10])).print()





