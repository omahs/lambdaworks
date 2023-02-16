use crate::elliptic_curve::traits::IsEllipticCurve;
use crate::field::element::FieldElement;
use std::fmt::Debug;

/// Represents an elliptic curve point using the projective short Weierstrass form:
/// y^2 * z = x^3 + a * x * z^2 + b * z^3,
/// where `x`, `y` and `z` variables are field elements.
#[derive(Debug, Clone)]
pub struct ProjectivePoint<E: IsEllipticCurve> {
    pub value: [FieldElement<E::BaseField>; 3],
}

impl<E: IsEllipticCurve> ProjectivePoint<E> {
    /// Creates an elliptic curve point giving the projective [x: y: z] coordinates.
    pub fn new(value: [FieldElement<E::BaseField>; 3]) -> Self {
        Self { value }
    }

    /// Returns the `x` coordinate of the point.
    pub fn x(&self) -> &FieldElement<E::BaseField> {
        &self.value[0]
    }

    /// Returns the `y` coordinate of the point.
    pub fn y(&self) -> &FieldElement<E::BaseField> {
        &self.value[1]
    }

    /// Returns the `z` coordinate of the point.
    pub fn z(&self) -> &FieldElement<E::BaseField> {
        &self.value[2]
    }

    /// Returns a tuple [x, y, z] with the coordinates of the point.
    pub fn coordinates(&self) -> &[FieldElement<E::BaseField>; 3] {
        &self.value
    }

    /// Creates the same point in affine coordinates. That is,
    /// returns [x / z: y / z: 1] where `self` is [x: y: z].
    /// Panics if `self` is the point at infinity.
    pub fn to_affine(&self) -> Self {
        let [x, y, z] = self.coordinates();
        assert_ne!(z, &FieldElement::zero());
        ProjectivePoint::new([x / z, y / z, FieldElement::one()])
    }
}

impl<E: IsEllipticCurve> PartialEq for ProjectivePoint<E> {
    fn eq(&self, other: &Self) -> bool {
        let [px, py, pz] = self.coordinates();
        let [qx, qy, qz] = other.coordinates();
        (px * qz == pz * qx) && (px * qy == py * qx)
    }
}

impl<E: IsEllipticCurve> Eq for ProjectivePoint<E> {}

#[cfg(test)]
mod tests {
    use crate::cyclic_group::IsGroup;
    use crate::elliptic_curve::short_weierstrass::curves::test_curve_1::{
        TestCurve1, TestCurveQuadraticNonResidue, TEST_CURVE_1_MAIN_SUBGROUP_ORDER,
        TEST_CURVE_1_PRIME_FIELD_ORDER,
    };
    use crate::elliptic_curve::short_weierstrass::curves::test_curve_2::TestCurve2;
    use crate::elliptic_curve::short_weierstrass::pairing::{tate_pairing, weil_pairing};
    use crate::field::element::FieldElement;
    use crate::field::fields::u64_prime_field::U64FieldElement;
    use crate::unsigned_integer::element::U384;
    //use crate::elliptic_curve::curves::test_curve_2::TestCurve2;
    use crate::elliptic_curve::traits::IsEllipticCurve;
    use crate::field::extensions::quadratic::QuadraticExtensionFieldElement;

    #[allow(clippy::upper_case_acronyms)]
    type FEE = QuadraticExtensionFieldElement<TestCurveQuadraticNonResidue>;

    // This tests only apply for the specific curve found in the configuration file.
    #[test]
    fn create_valid_point_works() {
        let point = TestCurve1::create_point_from_affine(FEE::from(35), FEE::from(31));
        assert_eq!(*point.x(), FEE::from(35));
        assert_eq!(*point.y(), FEE::from(31));
        assert_eq!(*point.z(), FEE::from(1));
    }

    #[test]
    #[should_panic]
    fn create_invalid_points_panics() {
        TestCurve1::create_point_from_affine(FEE::from(0), FEE::from(1));
    }

    #[test]
    fn equality_works() {
        let g = TestCurve1::generator();
        let g2 = g.operate_with(&g);
        assert_ne!(&g2, &g);
        assert_eq!(&g, &g);
    }

    #[test]
    fn operate_with_self_works_1() {
        let g = TestCurve1::generator();
        assert_eq!(g.operate_with(&g).operate_with(&g), g.operate_with_self(3));
    }

    #[test]
    fn operate_with_self_works_2() {
        let mut point_1 = TestCurve1::generator();
        point_1 = point_1.operate_with_self(TEST_CURVE_1_MAIN_SUBGROUP_ORDER as u128);
        assert!(point_1.is_neutral_element());
    }

    #[test]
    fn doubling_a_point_works() {
        let point = TestCurve1::create_point_from_affine(FEE::from(35), FEE::from(31));
        let expected_result = TestCurve1::create_point_from_affine(FEE::from(25), FEE::from(29));
        assert_eq!(point.operate_with_self(2).to_affine(), expected_result);
    }

    #[test]
    fn test_weil_pairing() {
        type FE = U64FieldElement<TEST_CURVE_1_PRIME_FIELD_ORDER>;
        let pa = TestCurve1::create_point_from_affine(FEE::from(35), FEE::from(31));
        let pb = TestCurve1::create_point_from_affine(
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
        );
        let expected_result = FEE::new([FE::new(46), FE::new(3)]);

        let result_weil = weil_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }

    #[test]
    fn test_tate_pairing() {
        type FE = U64FieldElement<TEST_CURVE_1_PRIME_FIELD_ORDER>;
        let pa = TestCurve1::create_point_from_affine(FEE::from(35), FEE::from(31));
        let pb = TestCurve1::create_point_from_affine(
            FEE::new([FE::new(24), FE::new(0)]),
            FEE::new([FE::new(0), FE::new(31)]),
        );
        let expected_result = FEE::new([FE::new(42), FE::new(19)]);

        let result_weil = tate_pairing(&pa, &pb);
        assert_eq!(result_weil, expected_result);
    }

    #[test]
    fn operate_with_self_works_with_test_curve_2() {
        let mut point_1 = TestCurve2::generator();
        point_1 = point_1.operate_with_self(15);

        let expected_result = TestCurve2::create_point_from_affine(
            FieldElement::new([
                FieldElement::new(U384::from("7b8ee59e422e702458174c18eb3302e17")),
                FieldElement::new(U384::from("1395065adef5a6a5457f1ea600b5a3e4fb")),
            ]),
            FieldElement::new([
                FieldElement::new(U384::from("e29d5b15c42124cd8f05d3c8500451c33")),
                FieldElement::new(U384::from("e836ef62db0a47a63304b67c0de69b140")),
            ]),
        );

        assert_eq!(point_1, expected_result);
    }

    #[test]
    fn coordinate_getters_work() {
        let x = FEE::from(35);
        let y = FEE::from(31);
        let z = FEE::from(1);
        let point = TestCurve1::create_point_from_affine(x.clone(), y.clone());
        let coordinates = point.coordinates();
        assert_eq!(&x, point.x());
        assert_eq!(&y, point.y());
        assert_eq!(&z, point.z());
        assert_eq!(x, coordinates[0]);
        assert_eq!(y, coordinates[1]);
        assert_eq!(z, coordinates[2]);
    }
}