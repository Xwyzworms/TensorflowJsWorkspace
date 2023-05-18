import * as tf from "@tensorflow/tfjs"

const features = tf.tensor([
    [-121, 47],
    [-121.2 , 46.5],
    [-122, 46.4],
    [-120.9 , 46.7]
])

const labels = tf.tensor([
    [200],
    [250],
    [215],
    [240]
]) 


const predictionPoints = tf.tensor([-121, 47])

let distance = tf.sqrt(tf.pow(tf.sub(features, predictionPoints),2).sum(1,true))
const K = 2
let someDistance = features
                    .sub(predictionPoints)
                    .pow(2)
                    .sum(1, true)
                    .pow(0.5)
                    .concat(labels, 1)

distance = tf.concat([distance, labels], 1).unstack().sort( (a,b) => {
    return a[0] < b[0] ? -1 : 1 
}).slice(0, K)
const tfDistance = tf.tensor(distance)
tfDistance.print()

