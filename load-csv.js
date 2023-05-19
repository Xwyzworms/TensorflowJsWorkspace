const { train } = require("@tensorflow/tfjs")
const fs = require("fs")
const _ = require("lodash")
const shuffleSeed =  require("shuffle-seed")

//TODO
// 1. Define the Requirements
//  1.1 Extract columns
//  1.2 Create temporary array for 
//     1.2.1 Training
//     1.2.1 Testing 

function extractDataBasedOnColumns(data, columNames) 
{
    const headers = _.first(data)

    const indexes = _.map(columNames, column => headers.indexOf(column))
    
    const extractedColumns = _.map(data, row=> _.pullAt(row, indexes)) // Pull only column contents

    return extractedColumns
}
function load_csv(filename, 
    {
        dataColumns = [],
        labelColumns = [],
        converters = {},
        shuffle = false,
        splitTest = false
    }
    
) 
{
    let dataset = fs.readFileSync(filename, {encoding : "utf-8"})
    dataset = _.map(dataset.split("\n"), row => row.split(",")) 
    dataset = _.dropRightWhile(dataset ,row => _.isEqual(row, [""]))
 
    const headers = _.first(dataset)

    dataset = _.map(dataset, (row, index) => {
        if (index === 0) 
        {
            return row
        }
        return _.map(row, (element, index) => {
            if(converters[headers[index]]) 
            {
                const converted = converters[headers[index]](element)
                return _.isNaN(converted) ? element : converted
            }
            // Convert the whole passed element into float
            const result = parseFloat(element.replace('"', ''))
            return _.isNaN(result) ? element : result 
        
        })
    })

    let labels = extractDataBasedOnColumns(dataset, labelColumns)
    dataset = extractDataBasedOnColumns(dataset, dataColumns)
    dataset.shift()// Remove the index
    labels.shift() 

    if (shuffle) {
        shuffleSeed.shuffle(dataset, "phrase")
        shuffleSeed.shuffle(labels, "phrase")
    }


    if (splitTest) {
        const trainSize = _.isNumber(splitTest) ? splitTest : Math.floor(dataset.length / 2);

        return  {
            features : dataset.slice(trainSize),
            labels : labels.slice(trainSize),
            testFeatures : dataset.slice(0,trainSize),
            testLabels : labels.slice(0,trainSize)
        }
    }
    else {
        return {
            features : dataset,
            labels : labels
        }
    }
}
mainDict = {
    shuffle : true,
    splitTest : 10,
    dataColumns : ["sqft_lot", "sqft_living"],
    labelColumns : ["price"]
}
console.log(load_csv("kc_house_data.csv", mainDict))