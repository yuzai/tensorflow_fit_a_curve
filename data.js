import * as tf from '@tensorflow/tfjs';

export function generateData(num, coeff, sigma = 0.04) {
    return tf.tidy(() => {
        const [a, b, c, d] = [
            tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
      tf.scalar(coeff.d)
        ]

        const x = tf.randomUniform([num], -1, 1);
        const y = a.mul(x.pow(tf.scalar(3)))
            .add(b.mul(x.square()))
            .add(c.mul(x))
            .add(d)
            .add(tf.randomNormal([num], 0, sigma));

        const ymin = y.min();
        const ymax = y.max();
        const yrange = ymax.sub(ymin);
        const yNormalized = y.sub(ymin).div(yrange);

        return {
            x,
            yNormalized
        };
    })
}