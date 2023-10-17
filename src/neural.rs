use ndarray::{arr2, Array1, Array2, Dim, Ix};
use std::f32::consts::E;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivsigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

struct NN<T> {
    input: Array1<T>,
    hidden: Array2<T>,
    output: Array1<T>,
}

struct Data<T> {
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
        let input = Array1::<f32>::zeros(input_count as usize);
        let hidden = Array2::<f32>::zeros((hidden_count as usize, input_count as usize));
        let output = Array1::<f32>::zeros(output_count as usize);
        NN {
            input,
            hidden,
            output,
        }
    }
    pub fn forward_propagate(&self, data: Vec<Data<f32>>) {}
}
