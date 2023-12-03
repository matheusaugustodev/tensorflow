async function run(){
    const csvUrl = '/data/iris.csv';
    // const trainingData = tf.data.csv(csvUrl, {
    //     columnConfigs: {
    //         species: {
    //             isLabel: true
    //         }
    //     }
    // });

    const trainingData = IRIS_DATA

    const numOfFeatures = (IRIS_CLASSES).length - 1;
    const numOfSamples = 150;
    const convertedData =
          trainingData.map(({xs, ys}) => {
              const labels = [
                    ys.species == "setosa" ? 1 : 0,
                    ys.species == "virginica" ? 1 : 0,
                    ys.species == "versicolor" ? 1 : 0
              ] 
              return{ xs: Object.values(xs), ys: Object.values(labels)};
          }).batch(10);
    
    //build the model
    
    const model = tf.sequential();
    
    //shape 
    model.add(tf.layers.dense({inputShape: [numOfFeatures], activation: "sigmoid", units: 5}))
    //three units since there are three classes
    model.add(tf.layers.dense({activation: "softmax", units: 3}));
    
    //
    
    model.compile({loss: "categoricalCrossentropy", optimizer: tf.train.adam(0.06)});
    
    //Training
    await model.fitDataset(convertedData, 
                     {epochs:100,
                      callbacks:{
                          onEpochEnd: async(epoch, logs) =>{
                              console.log("Epoch: " + epoch + " Loss: " + logs.loss);
                          }
                      }});
    
    // Test Cases:
    
    // Setosa
    //t testVal = tf.tensor2d([4.4, 2.9, 1.4, 0.2], [1, 4]);
    
    // Versicolor
    const testVal = tf.tensor2d([6.4, 3.2, 4.5, 1.5], [1, 4]);
    
    // Virginica
    // const testVal = tf.tensor2d([5.8,2.7,5.1,1.9], [1, 4]);
    
    const prediction = model.predict(testVal);
    const pIndex = tf.argMax(prediction, axis=1).dataSync();
    
    const classNames = ["Setosa", "Virginica", "Versicolor"];
    
    // alert(prediction)
    alert(classNames[pIndex])
    
}
run();




function dadosTreinamento() {
          
    // Colete os dados de treinamento
    const dadosTreinamento = [
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
      [10, 11, 12],
    ];

    // Crie o modelo de aprendizado de máquina
    const redeNeural = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(3, activation="softmax"),
    ]);

    // Treine o modelo
    redeNeural.fit(dadosTreinamento, epochs=100);

    // Use o modelo para fazer previsões
    const valor = 100;
    const resultado = redeNeural.predict([[valor]]);

    // Verifique se o valor está dentro do padrão
    if (resultado[0][0] > 0.5) {
      console.log("O valor está dentro do padrão.");
    } else {
      console.log("O valor está fora do padrão.");
    }

  }

  dadosTreinamento()