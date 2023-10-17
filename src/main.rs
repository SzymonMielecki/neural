mod network;
use ndarray::array;
use network::Perceptron;

fn main() {
    let start = std::time::Instant::now();
    let mut model = Perceptron::new(2, 2, 0.1);
    let data = array![
        (array![0.0, 0.0], array![0.0]),
        (array![0.0, 1.0], array![1.0]),
        (array![1.0, 0.0], array![1.0]),
        (array![1.0, 1.0], array![0.0]),
    ];
    model.fit(data.clone(), 2000, 100);

    for (input, target) in data.into_iter() {
        let res = model.predict(input.clone());
        println!(
            "Input1: {} Input2: {} Target: {} Predicted: {}",
            input[0], input[1], target, res
        );
    }
    println!("Time elapsed: {:?}", start.elapsed());
}
