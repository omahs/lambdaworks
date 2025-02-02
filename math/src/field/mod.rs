/// Implementation of FieldElement, a generic element of a field.
pub mod element;
/// Implementation of quadratic extensions of fields.
pub mod extensions;
/// Implementation of particular cases of fields.
pub mod fields;
/// Field for test purposes.
pub(crate) mod test_fields;
/// Common behaviour for field elements.
pub mod traits;
