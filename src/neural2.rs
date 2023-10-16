use std::f64::consts::E;
use std::{cell::RefCell, rc::Rc, rc::Weak};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

fn derivsigmoid(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn mse_loss(y: f64, y_pred: f64) -> f64 {
    (y - y_pred).powi(2)
}

pub struct Data {
    pub x: f64,
    pub y: f64,
    pub out: f64,
}

pub struct Neuron {
    pub bias: f64,
    pub prev: Vec<Weak<RefCell<Neuron>>>,
    pub next: Vec<Rc<RefCell<Neuron>>>,
}

impl Neuron {
    pub fn new() -> Rc<RefCell<Neuron>> {
        Rc::new(RefCell::new(Neuron {
            bias: 0.0,
            prev: Vec::new(),
            next: Vec::new(),
        }))
    }

    pub fn add_next(&mut self, next: Rc<RefCell<Neuron>>) {
        let weak_next = Rc::downgrade(&next);
        self.next.push(Rc::clone(&next));
        next.borrow_mut().add_prev(weak_next);
    }

    pub fn add_prev(&mut self, prev: Weak<RefCell<Neuron>>) {
        self.prev.push(prev);
    }
}

pub struct Trainer {
    data: Vec<Data>,
    in_neurons: Vec<Neuron>,
    calc_neurons: Vec<Neuron>,
    out_neuron: Neuron,
}

impl Trainer {
    pub fn new(
        data: Vec<Data>,
        in_neurons: Vec<Neuron>,
        calc_neurons: Vec<Neuron>,
        out_neuron: Neuron,
    ) -> Trainer {
        Trainer {
            data,
            in_neurons,
            calc_neurons,
            out_neuron,
        }
    }
}
