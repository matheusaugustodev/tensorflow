async function doIris() {
    const [ xTrain, yTrain, xTest, yTest] = getIrisData(.2)

    model = await trainModel(xTrain, yTrain, xTest, yTest)

    const input = tf.tensor2d([ 5.8, 2.7, 5.1, 1.9], [1, 4])
    const prediction = model.predict(input)

    alert(prediction)

    const predictionWithArgMax = model.predict(input).argMax(-1).dataSync()
    alert(predictionWithArgMax)

    const xData = xTest.dataSync()
    const yTrue = yTest.argMax(-1).dataSync()

    const predictions = await model.predict(xTest)
    const yPred = yTest.argMax(-1).dataSync()

    var correct = 0
    var wrong = 0

    for(var i=0; i<yTrue.length; i++) {
        if(yTrue[i] == yPred[i]) correct++
        else wrong++
    }

    alert(`Prection error rate: ${wrong / yTrue.length}`)
}

async function trainModel(xTrain, yTrain, xTest, yTest) {
    const model = tf.sequential()
    const learningRate = 0.01
    const numberOfEpochs = 40
    const optimizer = tf.train.adam(learningRate)

    model.add(tf.layers.dense({ units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]] }))
    model.add(tf.layers.dense({ units: 3, activation: 'softmax' }))

    model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy', metrics: ['accuracy']})

    await model.fit( xTrain, yTrain, {
        epochs: numberOfEpochs,
        validationData: [xTest, yTrain],
        callbacks: {
            onEpochEnd: async (epoch, logs) => {
                console.log(`Epoch ${epoch} Logs: ${logs.loss}`)
                await tf.nextFrame()
            }
        }
    })
    
    return model
}

doIris()