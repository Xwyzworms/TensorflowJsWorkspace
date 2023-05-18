import * as tf from "@tensorflow/tfjs"

const tensorA = tf.tensor(
    [[ 3, 4],
     [ 2, 8],
     [ 7, 10],
     [12, 22],
     [33, 41],
     [2, 1]]
    
    )

tensorA.slice([1, 0], [-1,1]).print() //  [Rows, columns] Starting index


const tensorAConcat = tf.tensor([
    [1,2,3],
    [4,5,6]
]) 

const tensorBConcat = tf.tensor([
    [7, 8, 9],
    [10, 11, 12]
]) 

const concatenated = tf.concat([tensorAConcat, tensorBConcat], 1)
concatenated.print()
console.log(concatenated.shape)

const jumpData = tf.tensor(
    [
        [70, 40, 73],
        [62, 53, 25],
        [61, 65, 54],
        [59, 34, 73]
    ]
) 

let jumpSum = jumpData.sum(1).expandDims(0)
const playerInfo = tf.tensor(
    [
        [1, 2, 3, 4],
        [182, 173, 186, 190]
    ]
    ) 


jumpSum = jumpSum.concat(playerInfo)
jumpSum.transpose().print()

