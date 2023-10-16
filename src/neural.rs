use rand::Rng;
use std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct Data {
    x: f64,
    y: f64,
    out: f64,
}

impl Data {
    pub fn new(x: i32, y: i32, out: i32) -> Data {
        Data {
            x: x as f64,
            y: y as f64,
            out: out as f64,
        }
    }
}

pub struct NN {
    weights: Vec<f64>,
    biases: Vec<f64>,
}

impl NN {
    pub fn new() -> NN {
        let mut rng = rand::thread_rng();
        NN {
            weights: vec![
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
                rng.gen(),
            ],
            biases: vec![rng.gen(), rng.gen(), rng.gen()],
        }
    }

    pub fn feed_forward(&self, x: Vec<f64>) -> f64 {
        let h1 = sigmoid(self.weights[0] * x[0] + self.weights[1] * x[1] + self.biases[0]);
        let h2 = sigmoid(self.weights[2] * x[0] + self.weights[3] * x[1] + self.biases[1]);
        sigmoid(self.weights[4] * h1 + self.weights[5] * h2 + self.biases[2])
    }

    pub fn train(&mut self, data: Vec<Data>, learn_rate: f64, epochs: i32) {
        let start = std::time::Instant::now();
        for _ in 0..epochs {
            for (data_slice) in data.iter() {
                let sum_h1 = self.weights[0] * data_slice.x
                    + self.weights[1] * data_slice.y
                    + self.biases[0];
                let h1 = sigmoid(sum_h1);
                let h1_der = derivsigmoid(sum_h1);

                let sum_h2 = self.weights[2] * data_slice.x
                    + self.weights[3] * data_slice.y
                    + self.biases[1];
                let h2 = sigmoid(sum_h2);
                let h2_der = derivsigmoid(sum_h2);

                let sum_o1 = self.weights[4] * h1 + self.weights[5] * h2 + self.biases[2];
                let o1 = sigmoid(sum_o1);
                let o1_der = derivsigmoid(sum_o1);

                let d_mse = -2.0 * (data_slice.out - o1);

                let d_w5 = h1 * o1_der;
                let d_w6 = h2 * o1_der;
                let d_b3 = o1_der;

                let d_h1 = self.weights[4] * o1_der;
                let d_h2 = self.weights[5] * o1_der;

                let d_w1 = data_slice.x * h1_der;
                let d_w2 = data_slice.y * h1_der;
                let d_b1 = h1_der;

                let d_w3 = data_slice.x * h2_der;
                let d_w4 = data_slice.y * h2_der;
                let d_b2 = h2_der;

                self.weights[0] -= learn_rate * d_mse * d_h1 * d_w1;
                self.weights[1] -= learn_rate * d_mse * d_h1 * d_w2;
                self.biases[0] -= learn_rate * d_mse * d_h1 * d_b1;

                self.weights[2] -= learn_rate * d_mse * d_h2 * d_w3;
                self.weights[3] -= learn_rate * d_mse * d_h2 * d_w4;
                self.biases[1] -= learn_rate * d_mse * d_h2 * d_b2;

                self.weights[4] -= learn_rate * d_mse * d_w5;
                self.weights[5] -= learn_rate * d_mse * d_w6;
                self.biases[2] -= learn_rate * d_mse * d_b3;
            }
        }
        let elapsed = start.elapsed();
        println!("Done in {} ms", elapsed.as_millis());
    }
    pub fn predict(&self, x: i32, y: i32) -> f64 {
        let res = self.feed_forward(vec![x as f64, y as f64]);
        println!("x: {}, y: {} -> {}", x, y, res);
        res
    }
}
