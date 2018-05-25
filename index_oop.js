import {
    Linear_Model
} from './linear_model';
import NNmodel from './nn_model';
import {
    generateData
} from './data';
import {
    plotData,
    plotCoeff,
    plotDataAndPredictions
} from './ui'
import * as tf from '@tensorflow/tfjs';


(async function () {
    async function liner_method() {
        const linear_model = new Linear_Model();

        const trueCoefficients = {
            a: -.8,
            b: -.2,
            c: .9,
            d: .5
        };
        const trainingData = generateData(100, trueCoefficients);
        await plotData('#data .plot', trainingData.x, trainingData.yNormalized);

        const predictionsBefore = linear_model.predict(trainingData.x);
        await plotDataAndPredictions('#random .plot', trainingData.x, trainingData.yNormalized, predictionsBefore);

        linear_model.fit(trainingData.x, trainingData.yNormalized);
        const predictionsAfter = linear_model.predict(trainingData.x);

        await plotDataAndPredictions('#trained .plot', trainingData.x, trainingData.yNormalized, predictionsAfter);

    }

    async function nn_method() {
        const nnmodel = new NNmodel({
            inputSize: 1,
            hiddenLayerSize: 6,
            outputSize: 1,
            learningRate: 0.1
        });

        const trueCoefficients = {
            a: -.8,
            b: -.2,
            c: .9,
            d: .5
        };
        const trainingData = generateData(100, trueCoefficients);
        const trainingData_nn = {
            x: trainingData.x.reshape([100, 1]),
            yNormalized: trainingData.yNormalized.reshape([100, 1])
        }
        await plotData('#data2 .plot', trainingData.x, trainingData.yNormalized);


        const predictionsBefore = nnmodel.predict(trainingData_nn.x);
        await plotDataAndPredictions('#random2 .plot', trainingData.x, trainingData.yNormalized, predictionsBefore.reshape([100]));

        nnmodel.fit(trainingData_nn.x, trainingData_nn.yNormalized, 200);
        const predictionsAfter = nnmodel.predict(trainingData_nn.x);
        await plotDataAndPredictions('#trained2 .plot', trainingData.x, trainingData.yNormalized, predictionsAfter.reshape([100]));
    }

    async function nn_model() {
        const model = tf.sequential();
        model.add(tf.layers.dense({
            units: 6,
            inputShape: [1],
            activation:'sigmoid'
        }));
        model.add(tf.layers.dense({
            units:1,
            activation:'sigmoid'
        }));
        model.compile({
            optimizer:tf.train.adam(0.1),
            loss:'meanSquaredError'
        })

        const trueCoefficients = {
            a: -.8,
            b: -.2,
            c: .9,
            d: .5
        };
        const trainingData = generateData(100, trueCoefficients);
        const trainingData_nn = {
            x: trainingData.x.reshape([100,1]),
            yNormalized: trainingData.yNormalized.reshape([100, 1])
        }
        await plotData('#data3 .plot', trainingData.x, trainingData.yNormalized);

        const predictionsBefore = model.predict(trainingData_nn.x);
        await plotDataAndPredictions('#random3 .plot', trainingData.x, trainingData.yNormalized, predictionsBefore);
        
        const h = await model.fit(trainingData_nn.x,trainingData_nn.yNormalized,{
            epochs:200,
            batchSize:100
        })

        const predictionsAfter = model.predict(trainingData_nn.x);
        await plotDataAndPredictions('#trained3 .plot', trainingData.x, trainingData.yNormalized, predictionsAfter);
    }

    liner_method();
    nn_method();
    nn_model();
})()
