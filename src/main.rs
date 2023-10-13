mod neural;
fn main() {
    let data = vec![
        neural::Data {
            x: 0.0,
            y: 0.0,
            out: 0.0,
        },
        neural::Data {
            x: 0.0,
            y: 1.0,
            out: 0.0,
        },
        neural::Data {
            x: 1.0,
            y: 0.0,
            out: 0.0,
        },
        neural::Data {
            x: 1.0,
            y: 1.0,
            out: 1.0,
        },
    ];
    let mut nn = neural::new_network(2, 2, 1);
    nn.train(4000, data)
}
