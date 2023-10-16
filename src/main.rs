mod neural;
mod neural_dist;
use {neural::Data, neural::NN};
use {neural_dist::Data as DataD, neural_dist::NN as NND};

fn main() {
    println!("------------------------------");
    let data_or: Vec<Data> = vec![
        Data::new(0, 0, 0),
        Data::new(0, 1, 1),
        Data::new(1, 0, 1),
        Data::new(1, 1, 1),
    ];
    let mut nn_or = NN::new();
    nn_or.train(data_or, 0.1, 1000000);
    nn_or.predict(0, 1);
    nn_or.predict(1, 0);
    nn_or.predict(0, 0);
    nn_or.predict(1, 1);

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
    let datad_or: Vec<DataD> = vec![
        DataD::new(0, 0, 0),
        DataD::new(0, 1, 1),
        DataD::new(1, 0, 1),
        DataD::new(1, 1, 1),
    ];
    let mut nnd_or = NND::new();
    nnd_or.train(datad_or, 0.1, 1000000);
    nnd_or.predict(0, 1);
    nnd_or.predict(1, 0);
    nnd_or.predict(0, 0);
    nnd_or.predict(1, 1);

    let datad_and: Vec<DataD> = vec![
        DataD::new(0, 0, 0),
        DataD::new(0, 1, 0),
        DataD::new(1, 0, 0),
        DataD::new(1, 1, 1),
    ];
    let mut nnd_and = NND::new();
    nnd_and.train(datad_and, 0.1, 1000000);
    nnd_and.predict(0, 1);
    nnd_and.predict(1, 0);
    nnd_and.predict(0, 0);
    nnd_and.predict(1, 1);
}
