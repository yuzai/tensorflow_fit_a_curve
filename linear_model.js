import {Model} from './model';
import * as tf from '@tensorflow/tfjs';

//y = weight[0]*x^3+weight[1]*x^2+weight[2]*x+biases

function random(){
    return (Math.random()-0.5)*2;
}

export class Linear_Model extends Model{
    constructor(){
        super();
        this.init();
    }
    init(){
        this.weights = [];
        this.weights[0] = tf.variable(tf.scalar(random()));
        this.weights[1] = tf.variable(tf.scalar(random()));
        this.weights[2] = tf.variable(tf.scalar(random()));
        this.bias = tf.variable(tf.scalar(random()));

        this.learningRate = 0.5;
        this.optimizer = tf.train.sgd(0.5);
    }
    predict(inputXs){
        return tf.tidy(()=>{
            return this.weights[0].mul(inputXs.pow(tf.scalar(3)))
                .add(this.weights[1].mul(inputXs.square()))
                .add(this.weights[2].mul(inputXs))
                .add(this.bias);
        })
    }
    train(inputXs,inputYs){
        this.optimizer.minimize(()=>{
            const predictedYs = this.predict(inputXs);
            return this.loss(predictedYs,inputYs);
        })
    }
    loss(predictedYs,inputYs){
        return predictedYs.sub(inputYs).square().mean();
    }
}