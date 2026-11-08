/// Generates doctests with a given name from a markdown file.
///
/// ## Usage
///
/// ```rust
/// rocm_include::doctest!(readme: "../README.md");
/// ```
#[macro_export]
macro_rules! doctest {
    ($($([$($feature:literal),+$(,)?])?$name:ident:$path:literal),+$(,)?) => {$(
        #[allow(dead_code)]
        $(#[cfg(all($(feature = $feature),+))])?
        #[doc = include_str!("../README.md")]
        fn $name() {}
    )*};
}
