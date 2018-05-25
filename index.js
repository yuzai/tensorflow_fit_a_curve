import * as tf from '@tensorflow/tfjs'
import {
    generateData
} from './data';
import {
    plotData,
    plotCoeff,
    plotDataAndPredictions
} from './ui'

//set up variables
const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));
//build a model
function predict(x) {
    return tf.tidy(() => {
        return a.mul(x.pow(tf.scalar(3)))
            .add(b.mul(x.square()))
            .add(c.mul(x))
            .add(d);
    });
};

function loss(predictions, labels) {
    const meanSquareError = predictions.sub(labels).square().mean();
    return meanSquareError;
}

function train(xs, ys, numIterations = 75) {
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);
    for (let i = 0; i < numIterations; i++) {
        optimizer.minimize(() => {
            const predsYs = predict(xs);
            return loss(predsYs, ys);
        })
    }
}

async function main() {
    const trueCoefficients = {
        a: -.8,
        b: -.2,
        c: .9,
        d: .5
    };
    const trainingData = generateData(100, trueCoefficients);
    plotCoeff('#data .coeff', trueCoefficients);
    await plotData('#data .plot', trainingData.x, trainingData.yNormalized);

    const a_random = await a.data();
    const b_random = await b.data();
    const c_random = await c.data();
    const d_random = await d.data();
    plotCoeff('#random .coeff', {
        a: a_random[0],
        b: b_random[0],
        c: c_random[0],
        d: d_random[0]
    });

    const predictionsBefore = predict(trainingData.x);
    await plotDataAndPredictions('#random .plot', trainingData.x, trainingData.yNormalized,predictionsBefore);

    train(trainingData.x, trainingData.yNormalized);
    const predictionsAfter = predict(trainingData.x);
    plotCoeff('#trained .coeff', {
        a: a.dataSync()[0],
        b: b.dataSync()[0],
        c: c.dataSync()[0],
        d: d.dataSync()[0],
    });
    await plotDataAndPredictions('#trained .plot', trainingData.x, trainingData.yNormalized,predictionsAfter);
}

main();
