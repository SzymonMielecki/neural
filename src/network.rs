use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::RandomExt;
use neural::{Floater, Matrice};
use num::Float;
use rand::distributions::Uniform;

// pub struct Perceptron {
//     w1: Array2<f32>,
//     b1: Array2<f32>,
//     w2: Array2<f32>,
//     b2: Array2<f32>,
//     z1: Array2<f32>,
//     a1: Array2<f32>,
//     z2: Array2<f32>,
//     learning_rate: f32,
// }

pub struct Perceptron<T: Floater, const INPUT: usize, const HIDDEN: usize> {
    w1: Matrice<T, INPUT, HIDDEN>,
    b1: Matrice<T, 1, HIDDEN>,
    w2: Matrice<T, HIDDEN, 1>,
    b2: Matrice<T, 1, 1>,
    z1: Matrice<T, 1, HIDDEN>,
    a1: Matrice<T, 1, HIDDEN>,
    z2: Matrice<T, 1, 1>,
    learning_rate: T,
}

impl<T: Floater, const INPUT: usize, const HIDDEN: usize> Perceptron<T, INPUT, HIDDEN> {
    pub fn new(input_size: usize, hidden_size: usize, learning_rate: f32) -> Self {
        Perceptron {
            w1: Matrice::random(),
            b1: Matrice::random(),
            w2: Matrice::random(),
            b2: Matrice::random(),
            z1: Matrice::zeros(),
            a1: Matrice::zeros(),
            z2: Matrice::zeros(),
            learning_rate: T::from(learning_rate).unwrap(),
        }
    }
    fn sigmoid(&self, x: T) -> T {
        1.0 / (1.0 + (-x).exp())
    }
    fn feed_forward(&mut self, x: &Matrice<T, HIDDEN, 1>) -> Array2<f32> {
        self.z1 = x.dot(self.w1).add(&self.b1);
        self.a1 = self.z1.mapv(|x| self.sigmoid(x));
        self.z2 = self.a1.dot(&self.w2) + &self.b2;
        self.z2.mapv(|x| self.sigmoid(x))
    }
    fn calculate_mse_loss(&self, pred: &Array2<f32>, target: &Array1<f32>) -> Array2<f32> {
        2.0 * (pred - target) / target.len() as f32
    }
    fn calculate_gradient(
        &self,
        pred: &Array2<f32>,
        target: &Array1<f32>,
    ) -> (Array2<f32>, Array2<f32>) {
        (
            self.calculate_mse_loss(pred, target),
            pred.mapv(|x| self.sigmoid(x)),
        )
    }
    fn update_weights(&mut self, gradient: (Array2<f32>, Array2<f32>), inputs: Array2<f32>) {
        let (mse_loss, sensitivity) = gradient;
        let mse_s = mse_loss.clone() * sensitivity.clone();

        let output_activation = &self.a1;
        let dj_dw2 = output_activation.t().dot(&mse_loss);
        let dj_db2 = (mse_loss * sensitivity).sum_axis(Axis(0));

        let hidden_activation_derivative =
            self.z1.mapv(|x| self.sigmoid(x) * (1.0 - self.sigmoid(x)));
        let inputs_len = inputs.len();
        let input_data_reshaped = inputs.into_shape((inputs_len, 1)).unwrap();
        let dj_dw1 = input_data_reshaped.dot(&mse_s.dot(&self.w2.t()))
            * hidden_activation_derivative.clone();
        let dj_db1 = (mse_s.dot(&self.w2.t()) * hidden_activation_derivative).sum();

        self.w2 = self.w2.clone() - self.learning_rate * dj_dw2;
        self.b2 = self.b2.clone() - self.learning_rate * dj_db2;
        self.w1 = self.w1.clone() - self.learning_rate * dj_dw1;
        self.b1 = self.b1.clone() - self.learning_rate * dj_db1;
    }
    pub fn fit(
        &mut self,
        data: Array1<(Array1<f32>, Array1<f32>)>,
        epochs: usize,
        print_loss_every: usize,
    ) {
        for epoch in 1..=epochs {
            for (inputs, targets) in data.iter() {
                let y = self.feed_forward(inputs);
                let gradient = self.calculate_gradient(&y, targets);
                self.update_weights(gradient, inputs.clone().into_shape((1, 2)).unwrap());

                if print_loss_every > 0 && epoch % print_loss_every == 0 {
                    println!("Epoch: {} Loss: {}", epoch, y);
                }
            }
        }
    }
    pub fn predict(&mut self, inputs: Array1<f32>) -> bool {
        *self.feed_forward(&inputs).get((0, 0)).unwrap() > 0.5
    }
}
