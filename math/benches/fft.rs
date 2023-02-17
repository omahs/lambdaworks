use const_random::const_random;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lambdaworks_math::fft::fft_cooley_tukey::{fft, inverse_fft};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::{IsField, IsTwoAdicField};
use rand::Rng;

// Copied from math/src/field/test_fields/u64_test_field.rs
#[derive(Debug, Clone, PartialEq, Eq)]
struct U64TestField<const MODULUS: u64>;

impl<const MODULUS: u64> IsField for U64TestField<MODULUS> {
    type BaseType = u64;

    fn add(a: &u64, b: &u64) -> u64 {
        ((*a as u128 + *b as u128) % MODULUS as u128) as u64
    }

    fn sub(a: &u64, b: &u64) -> u64 {
        (((*a as u128 + MODULUS as u128) - *b as u128) % MODULUS as u128) as u64
    }

    fn neg(a: &u64) -> u64 {
        MODULUS - a
    }

    fn mul(a: &u64, b: &u64) -> u64 {
        ((*a as u128 * *b as u128) % MODULUS as u128) as u64
    }

    fn div(a: &u64, b: &u64) -> u64 {
        Self::mul(a, &Self::inv(b))
    }

    fn inv(a: &u64) -> u64 {
        assert_ne!(*a, 0, "Cannot invert zero element");
        Self::pow(a, MODULUS - 2)
    }

    fn eq(a: &u64, b: &u64) -> bool {
        Self::from_u64(*a) == Self::from_u64(*b)
    }

    fn zero() -> u64 {
        0
    }

    fn one() -> u64 {
        1
    }

    fn from_u64(x: u64) -> u64 {
        x % MODULUS
    }

    fn from_base_type(x: u64) -> u64 {
        Self::from_u64(x)
    }
}

impl<const MODULUS: u64> IsTwoAdicField for U64TestField<MODULUS> {
    const TWO_ADICITY: u64 = 32;
    const TWO_ADIC_PRIMITVE_ROOT_OF_UNITY: u64 = 1753635133440165772;
    const GENERATOR: u64 = 7;
}

// Mersenne prime numbers
// https://www.math.utah.edu/~pa/math/mersenne.html
const PRIMES: [u64; 39] = [
    13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941,
    11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
    1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
];

fn fft_benchmark(c: &mut Criterion) {
    const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
    c.bench_function("fft", |bench| {
        let mut rng = rand::thread_rng();
        let coeffs_size = 1 << rng.gen_range(1..8);
        let mut coeffs: Vec<FieldElement<U64TestField<MODULUS>>> = vec![];

        for _ in 0..coeffs_size {
            coeffs.push(FieldElement::new(rng.gen_range(1..=u64::MAX)));
        }

        bench.iter(|| fft(black_box(&coeffs)));
    });
}

fn inverse_fft_benchmark(c: &mut Criterion) {
    const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
    c.bench_function("inverse_fft", |bench| {
        let mut rng = rand::thread_rng();
        let coeffs_size = 1 << rng.gen_range(1..8);
        let mut coeffs: Vec<FieldElement<U64TestField<MODULUS>>> = vec![];

        for _ in 0..coeffs_size {
            coeffs.push(FieldElement::new(rng.gen_range(1..=u64::MAX)));
        }

        let evaluations = fft(&coeffs).unwrap();

        bench.iter(|| inverse_fft(black_box(&evaluations)));
    });
}

criterion_group!(benches, fft_benchmark, inverse_fft_benchmark);
criterion_main!(benches);
