use crate::float::Float;
use ndarray::{Array2, Array3};
use serde::{Deserialize, Serialize};

/// Grid metadata for a 3D periodic domain.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GridSpec {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub lx: f64,
    pub ly: f64,
    pub lz: f64,
}

impl GridSpec {
    /// Create a cubic grid with side length `l` and `n` points per axis.
    pub fn cubic(n: usize, l: f64) -> Self {
        Self {
            nx: n,
            ny: n,
            nz: n,
            lx: l,
            ly: l,
            lz: l,
        }
    }

    /// Grid spacing (assumes uniform, uses x-direction).
    pub fn dx(&self) -> f64 {
        self.lx / self.nx as f64
    }

    pub fn dy(&self) -> f64 {
        self.ly / self.ny as f64
    }

    pub fn dz(&self) -> f64 {
        self.lz / self.nz as f64
    }

    /// Total number of physical-space grid points.
    pub fn total_points(&self) -> usize {
        self.nx * self.ny * self.nz
    }

    /// Spectral-space shape after R2C transform: (nx, ny, nz/2+1).
    pub fn spectral_shape(&self) -> (usize, usize, usize) {
        (self.nx, self.ny, self.nz / 2 + 1)
    }

    /// 3/2-padded grid for dealiasing.
    pub fn padded_3half(&self) -> Self {
        Self {
            nx: 3 * self.nx / 2,
            ny: 3 * self.ny / 2,
            nz: 3 * self.nz / 2,
            lx: self.lx,
            ly: self.ly,
            lz: self.lz,
        }
    }

    /// Volume element (uniform grid).
    pub fn dv(&self) -> f64 {
        self.dx() * self.dy() * self.dz()
    }
}

/// Grid metadata for an axisymmetric (r, z) domain.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AxiGridSpec {
    pub nr: usize,
    pub nz: usize,
    pub r_max: f64,
    pub lz: f64,
}

/// A 3D scalar field on a periodic grid.
#[derive(Debug, Clone)]
pub struct ScalarField<F: Float> {
    pub data: Array3<F>,
    pub grid: GridSpec,
}

impl<F: Float> ScalarField<F> {
    pub fn zeros(grid: GridSpec) -> Self {
        Self {
            data: Array3::from_elem((grid.nx, grid.ny, grid.nz), F::ZERO),
            grid,
        }
    }
}

/// A 3D vector field (3 components) on a periodic grid.
#[derive(Debug, Clone)]
pub struct VectorField<F: Float> {
    pub data: [Array3<F>; 3],
    pub grid: GridSpec,
}

impl<F: Float> VectorField<F> {
    pub fn zeros(grid: GridSpec) -> Self {
        let shape = (grid.nx, grid.ny, grid.nz);
        Self {
            data: [
                Array3::from_elem(shape, F::ZERO),
                Array3::from_elem(shape, F::ZERO),
                Array3::from_elem(shape, F::ZERO),
            ],
            grid,
        }
    }

    /// Convenience accessors.
    pub fn x(&self) -> &Array3<F> {
        &self.data[0]
    }
    pub fn y(&self) -> &Array3<F> {
        &self.data[1]
    }
    pub fn z(&self) -> &Array3<F> {
        &self.data[2]
    }
    pub fn x_mut(&mut self) -> &mut Array3<F> {
        &mut self.data[0]
    }
    pub fn y_mut(&mut self) -> &mut Array3<F> {
        &mut self.data[1]
    }
    pub fn z_mut(&mut self) -> &mut Array3<F> {
        &mut self.data[2]
    }
}

/// A 2D scalar field on an axisymmetric (r, z) grid.
#[derive(Debug, Clone)]
pub struct AxiScalarField<F: Float> {
    pub data: Array2<F>,
    pub grid: AxiGridSpec,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_spec_basic() {
        let g = GridSpec::cubic(64, 2.0 * std::f64::consts::PI);
        assert_eq!(g.nx, 64);
        assert_eq!(g.ny, 64);
        assert_eq!(g.nz, 64);
        let dx = g.dx();
        assert!((dx - 2.0 * std::f64::consts::PI / 64.0).abs() < 1e-14);
    }

    #[test]
    fn grid_spec_spectral_shape() {
        let g = GridSpec::cubic(64, 2.0 * std::f64::consts::PI);
        let (snx, sny, snz) = g.spectral_shape();
        assert_eq!(snx, 64);
        assert_eq!(sny, 64);
        assert_eq!(snz, 33); // N/2 + 1
    }

    #[test]
    fn grid_spec_padded() {
        let g = GridSpec::cubic(64, 2.0 * std::f64::consts::PI);
        let pg = g.padded_3half();
        assert_eq!(pg.nx, 96); // 3*64/2
        assert_eq!(pg.ny, 96);
        assert_eq!(pg.nz, 96);
    }

    #[test]
    fn scalar_field_zeros() {
        let g = GridSpec::cubic(8, 1.0);
        let s = ScalarField::<f64>::zeros(g);
        assert_eq!(s.data.shape(), &[8, 8, 8]);
        assert_eq!(s.data[[0, 0, 0]], 0.0);
    }

    #[test]
    fn vector_field_zeros() {
        let g = GridSpec::cubic(8, 1.0);
        let v = VectorField::<f64>::zeros(g);
        assert_eq!(v.x().shape(), &[8, 8, 8]);
        assert_eq!(v.y().shape(), &[8, 8, 8]);
        assert_eq!(v.z().shape(), &[8, 8, 8]);
    }

    #[test]
    fn total_points() {
        let g = GridSpec::cubic(32, 1.0);
        assert_eq!(g.total_points(), 32 * 32 * 32);
    }
}
