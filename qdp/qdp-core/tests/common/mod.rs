/// Create test data with normalized values
pub fn create_test_data(size: usize) -> Vec<f64> {
    (0..size).map(|i| (i as f64) / (size as f64)).collect()
}
