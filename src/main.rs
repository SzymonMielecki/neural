mod neural3;
use {neural3::DataIn, neural3::DataOut, neural3::NN};

fn main() {
    let data_in = vec![
        DataIn::new(0.0, 0.0),
        DataIn::new(0.0, 1.0),
        DataIn::new(1.0, 0.0),
        DataIn::new(1.0, 1.0),
    ];
    let data_out = vec![
        DataOut::new(0.0),
        DataOut::new(0.0),
        DataOut::new(0.0),
        DataOut::new(1.0),
    ];
    let mut nn = NN::new();

    nn.train(data_in, data_out, 0.1, 1000000);

    nn.predict(0, 1);
    nn.predict(1, 0);
    nn.predict(0, 0);
    nn.predict(1, 1);
}
