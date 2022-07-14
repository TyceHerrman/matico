use rusty_machine::linalg::Matrix;
use rusty_machine::learning::k_means::KMeansClassifier;
use rusty_machine::learning::UnSupModel;
use rusty_machine::prelude::BaseMatrix;

fn main() {
    let mut data = vec![-0.773196217050617, 0.24842717545639237,
    -0.6598113252414564, 0.6920640566349373,
    -0.23518920803371415, -0.5616678850149022,
    -0.2816950877136631, -0.9114944430563943,
    -0.24893149862052785, 0.584049927279119,
    0.7188483142673544, 0.4163443332288843,
    0.28795174508987703, 1.0276695211320594,
    -1.078385486977444, 0.8874191999016873,
    0.23384176150735006, -0.7151122736860034,
    -0.3481593622218171, 2.845586320877743];
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min_value = data[0];
    let max_value = data[data.len() - 1];
    
    // let minValue = &data.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap());
    // let maxValue = &data.into_iter().max_by(|a, b| a.partial_cmp(b).unwrap());

    let inputs = Matrix::new(20, 1, data);

    // Create model with k classes.
    let mut model = KMeansClassifier::new(5);

    // Where inputs is a Matrix with features in columns.
    model.train(&inputs).unwrap();
    let output = model.centroids().as_ref().unwrap(); //.as_ref(); //.transpose().get_row(0).unwrap();

    let mut centroids = vec![];
    for i in 0..output.rows() {
        centroids.push(output.get_row(i).unwrap()[0]);
    }
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("Centroids: {:#?}", centroids);

    let mut bounds = vec![];
    bounds.push(min_value);

    // for i in 0..centroids.len() - 1 {
    //     bounds.push(2.0*(centroids[i] as f64) - bounds[i] as f64);
    // }
    for i in 0..centroids.len() - 1 {
        bounds.push((centroids[i] as f64 + centroids[i + 1] as f64)/2.0);
    }
    bounds.push(max_value);
    println!("Bounds: {:#?}", bounds);

    // Where test_inputs is a Matrix with features in columns.
    // let a = model.predict(&test_inputs).unwrap();
}
