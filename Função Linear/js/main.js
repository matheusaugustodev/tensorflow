async function learnLinear() {
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))
    model.compile({
        loss: 'meanSquaredError',
        optimizer: 'sgd'
    })

    // Generate some synthetic data for training.
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1])
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1])

    await model.fit(xs, ys, { epochs: 1000 }).then(() => {
        model.predict(tf.tensor2d([10], [1, 1])).print()
    })
}

learnLinear()
