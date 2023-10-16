mod neural;
use {neural::Data, neural::NN};

fn main() {
    println!("------------------------------");
    let data_or: Vec<Data> = vec![
        Data::new(0, 0, 0),
        Data::new(0, 1, 1),
        Data::new(1, 0, 1),
        Data::new(1, 1, 1),
    ];
    let mut nn_or = NN::new();
    nn_or.train(data_or, 0.5, 1000000);
    nn_or.predict(0, 1);
    nn_or.predict(1, 0);
    nn_or.predict(0, 0);
    nn_or.predict(1, 1);
    println!("------------------------------");

    let data_and: Vec<Data> = vec![
        Data::new(0, 0, 0),
        Data::new(0, 1, 0),
        Data::new(1, 0, 0),
        Data::new(1, 1, 1),
    ];
    let mut nn_and = NN::new();
    nn_and.train(data_and, 0.1, 1000000);
    nn_and.predict(0, 1);
    nn_and.predict(1, 0);
    nn_and.predict(0, 0);
    nn_and.predict(1, 1);
    println!("------------------------------");
}
