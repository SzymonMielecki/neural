mod neural;
use ndarray::{arr1, arr2};
use neural::NN;

fn main() {
    println!("------------------------------");
    let data = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let res = arr1(&[0.0, 1.0, 1.0, 0.0]);
    let mut nn = NN::new(2, 2, 1);
    println!("------------------------------");
    nn.train(data, res, 0.1, 1000);
    println!("------------------------------");
    nn.predict(0, 1);
    nn.predict(1, 0);
    nn.predict(0, 0);
    nn.predict(1, 1);
    println!("------------------------------");
}
