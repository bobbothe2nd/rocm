use core::{
    fmt::{self, Display, Formatter},
    str::FromStr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GfxVersion {
    Gfx908,
    Gfx90a,
    Gfx90c,
    Gfx942,
    Gfx950,
    Gfx1030,
    Gfx1100,
    Gfx1101,
    Gfx1102,
    Gfx1103,
    Gfx1150,
    Gfx1151,
    Gfx1152,
    Gfx1153,
    Gfx1200,
    Gfx1201,
}

impl GfxVersion {
    pub const fn is_cdna(self) -> bool {
        matches!(
            self,
            Self::Gfx908 | Self::Gfx90a | Self::Gfx90c | Self::Gfx942 | Self::Gfx950
        )
    }

    pub const fn is_rdna(self) -> bool {
        !self.is_cdna()
    }

    pub const fn matrix_capabilities(self) -> MatrixCapabilities {
        match self {
            Self::Gfx908 => MatrixCapabilities {
                kind: MatrixKind::Mfma,
                wave_size: 64,
                operations: MatrixOperations::F16.union(MatrixOperations::F32),
            },

            Self::Gfx90a => MatrixCapabilities {
                kind: MatrixKind::Mfma,
                wave_size: 64,
                operations: MatrixOperations::F16
                    .union(MatrixOperations::BF16)
                    .union(MatrixOperations::F32)
                    .union(MatrixOperations::F64),
            },

            Self::Gfx942 | Self::Gfx950 => MatrixCapabilities {
                kind: MatrixKind::Mfma,
                wave_size: 64,
                operations: MatrixOperations::F16
                    .union(MatrixOperations::BF16)
                    .union(MatrixOperations::F32)
                    .union(MatrixOperations::F64)
                    .union(MatrixOperations::F8)
                    .union(MatrixOperations::BF8),
            },

            Self::Gfx1100 | Self::Gfx1101 | Self::Gfx1102 => MatrixCapabilities {
                kind: MatrixKind::Wmma,
                wave_size: 32,
                operations: MatrixOperations::F16
                    .union(MatrixOperations::BF16)
                    .union(MatrixOperations::F32),
            },

            Self::Gfx1200 | Self::Gfx1201 => MatrixCapabilities {
                kind: MatrixKind::Wmma,
                wave_size: 32,
                operations: MatrixOperations::F16
                    .union(MatrixOperations::BF16)
                    .union(MatrixOperations::F32)
                    .union(MatrixOperations::F8)
                    .union(MatrixOperations::BF8),
            },

            _ => MatrixCapabilities {
                kind: MatrixKind::None,
                wave_size: 0,
                operations: MatrixOperations::empty(),
            },
        }
    }

    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Gfx1030 => "gfx1030",
            Self::Gfx1103 => "gfx1103",
            Self::Gfx1150 => "gfx1150",
            Self::Gfx1151 => "gfx1151",
            Self::Gfx1152 => "gfx1152",
            Self::Gfx1153 => "gfx1153",
            Self::Gfx908 => "gfx908",
            Self::Gfx90a => "gfx90a",
            Self::Gfx90c => "gfx90c",
            Self::Gfx942 => "gfx942",
            Self::Gfx950 => "gfx950",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1101 => "gfx1101",
            Self::Gfx1102 => "gfx1102",
            Self::Gfx1200 => "gfx1200",
            Self::Gfx1201 => "gfx1201",
        }
    }
}

impl Display for GfxVersion {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for GfxVersion {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let head = s.trim().split(':').next().ok_or(())?;
        let head = {
            #[cfg(not(feature = "alloc"))]
            {
                head
            }

            #[cfg(feature = "alloc")]
            {
                head.to_lowercase()
            }
        };
        let head = {
            #[cfg(not(feature = "alloc"))]
            {
                head
            }

            #[cfg(feature = "alloc")]
            {
                head.as_str()
            }
        };

        let str = match head {
            "gfx1030" => Self::Gfx1030,
            "gfx1103" => Self::Gfx1103,
            "gfx1150" => Self::Gfx1150,
            "gfx1151" => Self::Gfx1151,
            "gfx1152" => Self::Gfx1152,
            "gfx1153" => Self::Gfx1153,
            "gfx908" => Self::Gfx908,
            "gfx90a" => Self::Gfx90a,
            "gfx90c" => Self::Gfx90c,
            "gfx942" => Self::Gfx942,
            "gfx950" => Self::Gfx950,
            "gfx1100" => Self::Gfx1100,
            "gfx1101" => Self::Gfx1101,
            "gfx1102" => Self::Gfx1102,
            "gfx1200" => Self::Gfx1200,
            "gfx1201" => Self::Gfx1201,
            _ => return Err(()),
        };

        Ok(str)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatrixCapabilities {
    pub kind: MatrixKind,
    pub wave_size: u32,
    pub operations: MatrixOperations,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixKind {
    None,
    Mfma,
    Wmma,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct MatrixOperations: u32 {
        const F16  = 1 << 0;
        const BF16 = 1 << 1;
        const F32  = 1 << 2;
        const F64  = 1 << 3;
        const F8   = 1 << 4;
        const BF8  = 1 << 5;
    }
}
