use ndarray::{arr1, arr2, Array1, Array2, Dim, Ix};
use std::f32::consts::E;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivsigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct NN<T> {
    input: InputLayer<T>,
    hidden: HiddenLayer<T>,
    output: OutputLayer<T>,
}

struct InputLayer<T> {
    input: Array1<T>,
}

impl InputLayer<f32> {
    pub fn new(input_count: i32) -> Self {
        let input = Array1::<f32>::zeros(input_count as usize);
        InputLayer { input }
    }
}

struct HiddenLayer<T> {
    weights: Array2<T>,
    biases: Array1<T>,
}

impl HiddenLayer<f32> {
    pub fn new(input_count: i32, hidden_count: i32) -> Self {
        let weights = Array2::<f32>::zeros((hidden_count as usize, input_count as usize));
        let biases = Array1::<f32>::zeros(hidden_count as usize);
        HiddenLayer { weights, biases }
    }
}

struct OutputLayer<T> {
    weights: Array2<T>,
    biases: Array1<T>,
}

impl OutputLayer<f32> {
    pub fn new(hidden_count: i32, output_count: i32) -> Self {
        let weights = Array2::<f32>::zeros((output_count as usize, hidden_count as usize));
        let biases = Array1::<f32>::zeros(output_count as usize);
        OutputLayer { weights, biases }
    }
}

pub struct Data<T> {
    x: T,
    y: T,
    out: T,
}

impl Data<f32> {
    pub fn new(x: f32, y: f32, out: f32) -> Self {
        Data { x, y, out }
    }
}

impl NN<f32> {
    pub fn new(input_count: i32, hidden_count: i32, output_count: i32) -> Self {
        let input = InputLayer::new(input_count); // 3
        let hidden = HiddenLayer::new(input_count, hidden_count);
        let output = OutputLayer::new(hidden_count, output_count);
        NN {
            input,
            hidden,
            output,
        }
    }
    pub fn feed_forward(&self, data: Array1<f32>) -> f32 {
        let h1 = sigmoid(self.hidden.weights.column(0).dot(&data) + self.hidden.biases[0]);
        let h2 = sigmoid(self.hidden.weights.column(1).dot(&data) + self.hidden.biases[1]);
        let o1 =
            sigmoid(self.output.weights.column(0).dot(&arr1(&[h1, h2])) + self.output.biases[0]);
        o1
    }
    pub fn train(&mut self, data: Array2<f32>, res: Array1<f32>, learn_rate: f32, epochs: usize) {
        let start = std::time::Instant::now();

        for _ in 0..epochs {
            for (data_slice, res_slice) in data.rows().into_iter().zip(res.iter()) {
                let sum_h = self.hidden.weights.dot(&data_slice) + &self.hidden.biases;
                let h = sum_h.mapv(sigmoid);
                let h_der = sum_h.mapv(derivsigmoid);

                let sum_o = self.output.weights.dot(&h) + self.output.biases[0];
                let o = sum_o.mapv(sigmoid);
                let o_der = sum_o.mapv(derivsigmoid);

                let d_mse = -2.0 * (&arr1(&[res_slice]) - o);

                let d_w_h = &o_der * &h;
                let d_b_h = o_der;

                let d_h = data_slice.dot(&self.output.weights.t()); // here
                                                                    // let d_h = self.output.weights.dot(&data_slice.t()); // here

                let d_w_i = &h_der * &d_h;
                let d_b_i = h_der;
                // println!("d_w_i: {:?}, d_b_i: {:?}", d_w_i, d_b_i);

                self.output.weights = &self.output.weights - learn_rate * &d_mse * &d_h * &d_w_h;
                self.output.biases = &self.output.biases - learn_rate * &d_mse * &d_h * &d_b_h;
                // println!("output.weights: {:?}", self.output.weights);
                // println!("output.biases: {:?}", self.output.biases);

                self.hidden.weights =
                    &self.hidden.weights - learn_rate * &d_mse * data_slice * &d_w_i;
                self.hidden.biases =
                    &self.hidden.biases - learn_rate * &d_mse * data_slice * &d_b_i;
                // println!("hidden.weights: {:?}", self.hidden.weights);
                // println!("hidden.biases: {:?}", self.hidden.biases);
            }
        }
        println!("Time: {:?}", start.elapsed());
    }
    pub fn predict(&self, x: i32, y: i32) -> f32 {
        println!("------------------------------");
        let res = self.feed_forward(arr1(&[x as f32, y as f32]));

        println!("x: {}, y: {} -> {}", x, y, res);
        res
    }
}
