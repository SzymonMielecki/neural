use std::ops::{Add, Div, Mul};
pub struct Matrice<f32, const COLS: usize, const ROWS: usize>([[f32; COLS]; ROWS]);

impl<f32: Mul<Output = f32> + Copy, const COLS: usize, const ROWS: usize> Mul<f32>
    for Matrice<f32, COLS, ROWS>
{
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        let mut result = Matrice::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.0[i][j] = self.0[i][j] * rhs;
            }
        }
        result
    }
}

impl<f32: Div<Output = f32>, const COLS: usize, const ROWS: usize> Div<f32>
    for Matrice<f32, COLS, ROWS>
{
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        let mut result = self;
        for i in 0..ROWS {
            for j in 0..COLS {
                result.0[i][j] = result.0[i][j] / rhs;
            }
        }
        result
    }
}

trait Zero {
    fn zero() -> Self;
}

impl Zero for f32 {
    fn zero() -> Self {
        0.0
    }
}

trait Rand {
    fn rand() -> Self;
}

impl Rand for f32 {
    fn rand() -> Self {
        rand::random()
    }
}

impl<
        f32: std::marker::Copy + Add<Output = f32> + Mul<Output = f32> + Zero + Rand,
        const COLS: usize,
        const ROWS: usize,
    > Matrice<f32, COLS, ROWS>
{
    pub fn zeros() -> Self {
        Matrice([[f32::zero(); COLS]; ROWS])
    }
    pub fn random() -> Self {
        Matrice([[f32::rand(); COLS]; ROWS])
    }
    pub fn columns(&self) -> usize {
        COLS
    }
    pub fn rows(&self) -> usize {
        ROWS
    }
    pub fn dot<const COLS1: usize, const ROWS1: usize>(
        &self,
        dotter: Matrice<f32, COLS1, ROWS1>,
    ) -> Matrice<f32, ROWS, COLS> {
        if self.columns() != dotter.rows() {
            panic!("Matrice dimensions do not match");
        }
        let mut result = Matrice::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.0[i][j] = self.0[i][j] * dotter.0[j][i];
            }
        }
        result
    }
    pub fn add(&self, adder: &Matrice<f32, COLS, ROWS>) -> Matrice<f32, COLS, ROWS> {
        let mut result = Matrice::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.0[i][j] = self.0[i][j] + adder.0[i][j];
            }
        }
        result
    }
    pub fn mapv(&self, f: impl Fn(f32) -> f32) -> Matrice<f32, COLS, ROWS> {
        let mut result = Matrice::zeros();
        for i in 0..ROWS {
            for j in 0..COLS {
                result.0[i][j] = f(self.0[i][j]);
            }
        }
        result
    }
}
