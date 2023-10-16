use rand::Rng;
use std::f32::consts::E;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivsigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct Data {
    x: f32,
    y: f32,
    out: f32,
}

impl Data {
    pub fn new(x: i32, y: i32, out: i32) -> Data {
        Data {
            x: x as f32,
            y: y as f32,
            out: out as f32,
        }
    }
}

pub struct NN {
    conns: Conns,
    neurons: Neurons,
}

struct Conn {
    weight: f32,
}

impl Conn {
    pub fn new() -> Conn {
        let mut rng = rand::thread_rng();
        Conn { weight: rng.gen() }
    }
}

struct Conns {
    w0: Conn,
    w2: Conn,
    w1: Conn,
    w3: Conn,
    w4: Conn,
    w5: Conn,
}

impl Conns {
    pub fn new() -> Conns {
        let mut rng = rand::thread_rng();
        Conns {
            w0: Conn::new(),
            w2: Conn::new(),
            w1: Conn::new(),
            w3: Conn::new(),
            w4: Conn::new(),
            w5: Conn::new(),
        }
    }
}

struct Neuron {
    bias: f32,
}

impl Neuron {
    pub fn new() -> Neuron {
        let mut rng = rand::thread_rng();
        Neuron { bias: rng.gen() }
    }
}

struct Neurons {
    x0: Neuron,
    x1: Neuron,
    h0: Neuron,
    h1: Neuron,
    h2: Neuron,
    o0: Neuron,
}

impl Neurons {
    pub fn new() -> Neurons {
        let mut rng = rand::thread_rng();
        Neurons {
            x0: Neuron::new(),
            x1: Neuron::new(),
            h0: Neuron::new(),
            h1: Neuron::new(),
            h2: Neuron::new(),
            o0: Neuron::new(),
        }
    }
}

impl NN {
    pub fn new() -> NN {
        let mut rng = rand::thread_rng();
        NN {
            conns: Conns::new(),
            neurons: Neurons::new(),
        }
    }

    pub fn feed_forward(&self, x0: f32, x1: f32) -> f32 {
        let h1 =
            sigmoid(self.conns.w0.weight * x0 + self.conns.w1.weight * x1 + self.neurons.x0.bias);
        let h2 =
            sigmoid(self.conns.w2.weight * x0 + self.conns.w3.weight * x1 + self.neurons.x1.bias);
        sigmoid(self.conns.w4.weight * h1 + self.conns.w5.weight * h2 + self.neurons.o0.bias)
    }

    pub fn train(&mut self, data: Vec<Data>, learn_rate: f32, epochs: i32) {
        let start = std::time::Instant::now();
        for _ in 0..epochs {
            for (data_slice) in data.iter() {
                let sum_h1 = self.conns.w0.weight * data_slice.x
                    + self.conns.w1.weight * data_slice.y
                    + self.neurons.h1.bias;
                let h1 = sigmoid(sum_h1);
                let h1_der = derivsigmoid(sum_h1);

                let sum_h2 = self.conns.w2.weight * data_slice.x
                    + self.conns.w3.weight * data_slice.y
                    + self.neurons.h2.bias;
                let h2 = sigmoid(sum_h2);
                let h2_der = derivsigmoid(sum_h2);

                let sum_o0 =
                    self.conns.w4.weight * h1 + self.conns.w5.weight * h2 + self.neurons.o0.bias;
                let o0 = sigmoid(sum_o0);
                let o0_der = derivsigmoid(sum_o0);

                let d_mse = -2.0 * (data_slice.out - o0);

                let d_w5 = h1 * o0_der;
                let d_w6 = h2 * o0_der;
                let d_b3 = o0_der;

                let d_h1 = self.conns.w4.weight * o0_der;
                let d_h2 = self.conns.w5.weight * o0_der;

                let d_w1 = data_slice.x * h1_der;
                let d_w2 = data_slice.y * h1_der;
                let d_b1 = h1_der;

                let d_w3 = data_slice.x * h2_der;
                let d_w4 = data_slice.y * h2_der;
                let d_b2 = h2_der;

                self.conns.w0.weight -= learn_rate * d_mse * d_h1 * d_w1;
                self.conns.w1.weight -= learn_rate * d_mse * d_h1 * d_w2;
                self.neurons.h1.bias -= learn_rate * d_mse * d_h1 * d_b1;

                self.conns.w2.weight -= learn_rate * d_mse * d_h2 * d_w3;
                self.conns.w3.weight -= learn_rate * d_mse * d_h2 * d_w4;
                self.neurons.h2.bias -= learn_rate * d_mse * d_h2 * d_b2;

                self.conns.w4.weight -= learn_rate * d_mse * d_w5;
                self.conns.w5.weight -= learn_rate * d_mse * d_w6;
                self.neurons.o0.bias -= learn_rate * d_mse * d_b3;
            }
        }
        let elapsed = start.elapsed();
        println!("Done in {} ms", elapsed.as_millis());
    }
    pub fn predict(&self, x: i32, y: i32) -> f32 {
        let res = self.feed_forward(x as f32, y as f32);
        println!("x: {}, y: {} -> {}", x, y, res);
        res
    }
}
