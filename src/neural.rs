use rand::Rng;
use std::f32::consts::E;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivsigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn mse_loss(y: f32, y_pred: f32) -> f32 {
    (y - y_pred).powi(2)
}

struct InputNeuron {
    data: f32,
}
struct InputCalcConn {
    in_neuron: InputNeuron,
    weight: f32,
}

struct CalcNeuron {
    in_conn: Vec<InputCalcConn>,
    data: f32,
    bias: f32,
}
struct CalcOutputConn {
    in_neuron: CalcNeuron,
    weight: f32,
}
struct OutputNeuron {
    in_conn: Vec<CalcOutputConn>,
    data: f32,
    bias: f32,
}

impl CalcNeuron {
    pub fn calc_val(&self) -> f32 {
        let mut sum = 0.0;
        self.in_conn
            .iter()
            .for_each(|conn| sum += conn.in_neuron.data * conn.weight);
        sigmoid(sum + self.bias)
    }
    pub fn calc_loss(&self, y: f32) -> f32 {
        mse_loss(y, self.data)
    }
    pub fn calc_conn_weight_delta(&self, y_pred: f32) -> Vec<f32> {
        let mut delta: Vec<f32> = Vec::with_capacity(self.in_conn.len());
        self.in_conn
            .iter()
            .for_each(|conn| delta.push(conn.in_neuron.data * self.calc_loss(y_pred)));
        delta
    }
    pub fn calc_bias(&self) -> f32 {
        derivsigmoid(self.data)
    }
    pub fn update_neuron_and_conns(&mut self, learning_rate: f32, y_pred: f32) {
        self.data = self.calc_val();
        self.bias = self.calc_bias();
        self.in_conn.iter_mut().for_each(|conn| {
            conn.weight -= learning_rate
                * conn.in_neuron.data
                * derivsigmoid(self.data)
                * mse_loss(self.data, y_pred)
        });
    }
}

impl OutputNeuron {
    pub fn calc_val(&self) -> f32 {
        let mut sum = 0.0;
        self.in_conn
            .iter()
            .for_each(|conn| sum += conn.in_neuron.data * conn.weight);
        sigmoid(sum + self.bias)
    }
    pub fn calc_loss(&self, y: f32) -> f32 {
        mse_loss(y, self.data)
    }
    pub fn calc_conn_weight_delta(&self, y_pred: f32) -> Vec<f32> {
        let mut delta: Vec<f32> = Vec::with_capacity(self.in_conn.len());
        self.in_conn
            .iter()
            .for_each(|conn| delta.push(conn.in_neuron.data * self.calc_loss(y_pred)));
        delta
    }
    pub fn calc_bias(&self) -> f32 {
        derivsigmoid(self.data)
    }
    pub fn update_neuron_and_conns(&mut self, learning_rate: f32, y_pred: f32) {
        self.data = self.calc_val();
        self.bias = self.calc_bias();
        self.in_conn.iter_mut().for_each(|conn| {
            conn.weight -= learning_rate
                * conn.in_neuron.data
                * derivsigmoid(self.data)
                * mse_loss(self.data, y_pred)
        });
    }
}

pub struct Data {
    pub x: f32,
    pub y: f32,
    pub out: f32,
}

pub struct Network {
    output: OutputNeuron,
    learning_rate: f32,
}

impl Network {
    pub fn feed_forward(&mut self, y_pred: f32) {
        self.output
            .update_neuron_and_conns(self.learning_rate, y_pred);
        for conn in &mut self.output.in_conn {
            conn.in_neuron
                .update_neuron_and_conns(self.learning_rate, y_pred);
        }
    }
    pub fn train(&mut self, epochs: usize, data: Vec<Data>) {
        for i in 0..=epochs {
            for data_slice in &data {
                self.feed_forward(data_slice.out);
            }
            if i % 1000 == 0 {
                println!("Epoch: {}", i);
                for data_slice in &data {
                    println!("Input: {}, {}", data_slice.x, data_slice.y);
                    println!("Prediction: {}", data_slice.out);
                    println!("Output: {}", self.output.data);
                }
            }
        }
    }
}

pub fn new_network(input_count: usize, calc_count: usize, learning_rate: i32) -> Network {
    let mut rng = rand::thread_rng();
    let mut net = Network {
        output: OutputNeuron {
            in_conn: Vec::with_capacity(calc_count),
            data: 0.0,
            bias: 0.0,
        },
        learning_rate: learning_rate as f32,
    };
    for _ in 0..calc_count {
        let mut conn = CalcOutputConn {
            in_neuron: CalcNeuron {
                in_conn: Vec::with_capacity(calc_count),
                data: 0.0,
                bias: 0.0,
            },
            weight: rng.gen_range(0.0..=1.0),
        };
        for _ in 0..input_count {
            conn.in_neuron.in_conn.push(InputCalcConn {
                in_neuron: InputNeuron { data: 0.0 },
                weight: rng.gen_range(0.0..=1.0),
            })
        }
        net.output.in_conn.push(conn);
    }
    net
}
