import {
    Model
} from './model';
import * as tf from '@tensorflow/tfjs';

function tensor(obj) {
    if (obj instanceof tf.Tensor) {
        return obj;
    }
    if (typeof obj === 'number') {
        return tf.scalar(obj);
    } else if (Array.isArray(obj)) {
        return tf.tensor(obj);
    }
    throw new Error(
        'tensor() only supports number or array as the input parameter.'
    );
}

//generatic nn network

export default class NNModel extends Model {
    constructor({
        inputSize = 3,
        hiddenLayerSize = inputSize * 2,
        outputSize = 2,
        learningRate = 0.1
    } = {}) {
        super();
        this.hiddenLayerSize = hiddenLayerSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.optimizer = tf.train.adam(learningRate);
        this.init();
    }

    init() {
        this.weights = [];
        this.biases = [];
        this.weights[0] = tf.variable(
            tf.randomNormal([this.inputSize, this.hiddenLayerSize])
        );
        this.biases[0] = tf.variable(tf.scalar(Math.random()));
        // Output layer
        this.weights[1] = tf.variable(
            tf.randomNormal([this.hiddenLayerSize, this.outputSize])
        );
        this.biases[1] = tf.variable(tf.scalar(Math.random()));
    }

    predict(inputXs) {
        const x = tensor(inputXs);
        return tf.tidy(()=>{
            const hiddenLayer = tf.sigmoid(x.matMul(this.weights[0]).add(this.biases[0]));
            const outputLayer = tf.sigmoid(hiddenLayer.matMul(this.weights[1]).add(this.biases[1]));
            return outputLayer;
        })
    }

    // init(){
    //     this.weights = [];
    //     this.weights[0] = tf.variable(
    //         tf.randomNormal([this.inputSize+1,this.hiddenLayerSize])
    //     );
    //     this.weights[1] = tf.variable(
    //         tf.randomNormal([this.hiddenLayerSize+1,this.outputSize])
    //     );
    // }
    // predict(inputXs){
    //     const x = tf.concat([tensor(inputXs),tf.ones([100,1])],1);
    //     return tf.tidy(()=>{
    //         const hiddenLayer = tf.concat([tf.sigmoid(x.matMul(this.weights[0])),tf.ones([100,1])],1);
    //         const outputLayer = tf.sigmoid(hiddenLayer.matMul(this.weights[1]));
    //         return outputLayer;
    //     })
    // }
    train(inputXs,inputYs){
        this.optimizer.minimize(()=>{
            const predictedYs = this.predict(inputXs);
            return  this.loss(predictedYs,inputYs);
        })
    }
    loss(predictedYs, inputYs) {
        const meanSquareError = predictedYs
          .sub(tensor(inputYs))
          .square()
          .mean();
        return meanSquareError;
      }
}