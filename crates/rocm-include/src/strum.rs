#[macro_export]
macro_rules! strum {
    ($($(#[$attr:meta])* $vis:vis enum $name:ident {
        $($var:ident=$str:literal),*$(,)?
    }; $err:ident;)+) => {$(
        $(#[$attr])*
        $vis enum $name {$(
            $var,
        )*}
        impl $name {
            pub const fn as_str(&self) -> &'static ::core::primitive::str {
                match self {$(Self::$var=>$str,)*}
            }
        }
        impl ::core::fmt::Display for $name {
            fn fmt(&self, f: &mut ::core::fmt::Formatter<'_>) -> ::core::fmt::Result {
                f.write_str(self.as_str())
            }
        }
        impl From<$name> for &'static ::core::primitive::str {
            fn from(value: $name) -> Self { value.as_str() }
        }
        $vis struct $err;
        impl ::core::str::FromStr for $name {
            type Err = $err;
            fn from_str(s: &::core::primitive::str) -> Result<Self, Self::Err> {
                let head = s.trim().split(':').next().ok_or($err)?;
                let str = match head {$($str=>Self::$var,)*_ => return Err($err)};
                Ok(str)
            }
        }
    )+};
}
