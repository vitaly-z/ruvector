//! Semantic versioning for apps

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;
use crate::error::{AppStoreError, AppStoreResult};

/// Semantic version
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub prerelease: Option<String>,
    pub build: Option<String>,
}

impl Version {
    /// Create a new version
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Version {
            major,
            minor,
            patch,
            prerelease: None,
            build: None,
        }
    }

    /// Parse version from string
    pub fn parse(s: &str) -> AppStoreResult<Self> {
        let s = s.trim().trim_start_matches('v');

        // Split build metadata
        let (version_pre, build) = if let Some(idx) = s.find('+') {
            (&s[..idx], Some(s[idx + 1..].to_string()))
        } else {
            (s, None)
        };

        // Split prerelease
        let (version, prerelease) = if let Some(idx) = version_pre.find('-') {
            (&version_pre[..idx], Some(version_pre[idx + 1..].to_string()))
        } else {
            (version_pre, None)
        };

        // Parse version numbers
        let parts: Vec<&str> = version.split('.').collect();
        if parts.len() < 2 || parts.len() > 3 {
            return Err(AppStoreError::InvalidApp(format!("Invalid version: {}", s)));
        }

        let major = parts[0].parse()
            .map_err(|_| AppStoreError::InvalidApp(format!("Invalid major version: {}", parts[0])))?;
        let minor = parts[1].parse()
            .map_err(|_| AppStoreError::InvalidApp(format!("Invalid minor version: {}", parts[1])))?;
        let patch = if parts.len() > 2 {
            parts[2].parse()
                .map_err(|_| AppStoreError::InvalidApp(format!("Invalid patch version: {}", parts[2])))?
        } else {
            0
        };

        Ok(Version {
            major,
            minor,
            patch,
            prerelease,
            build,
        })
    }

    /// Create version with prerelease
    pub fn with_prerelease(mut self, prerelease: &str) -> Self {
        self.prerelease = Some(prerelease.to_string());
        self
    }

    /// Create version with build metadata
    pub fn with_build(mut self, build: &str) -> Self {
        self.build = Some(build.to_string());
        self
    }

    /// Check if this is a prerelease version
    pub fn is_prerelease(&self) -> bool {
        self.prerelease.is_some()
    }

    /// Check if this version is compatible with another (same major version)
    pub fn is_compatible_with(&self, other: &Version) -> bool {
        if self.major == 0 && other.major == 0 {
            // For 0.x versions, minor version must match
            self.minor == other.minor
        } else {
            self.major == other.major
        }
    }

    /// Increment major version
    pub fn bump_major(&self) -> Version {
        Version::new(self.major + 1, 0, 0)
    }

    /// Increment minor version
    pub fn bump_minor(&self) -> Version {
        Version::new(self.major, self.minor + 1, 0)
    }

    /// Increment patch version
    pub fn bump_patch(&self) -> Version {
        Version::new(self.major, self.minor, self.patch + 1)
    }
}

impl Default for Version {
    fn default() -> Self {
        Version::new(0, 1, 0)
    }
}

impl fmt::Display for Version {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(ref pre) = self.prerelease {
            write!(f, "-{}", pre)?;
        }
        if let Some(ref build) = self.build {
            write!(f, "+{}", build)?;
        }
        Ok(())
    }
}

impl Ord for Version {
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare major.minor.patch
        match self.major.cmp(&other.major) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.minor.cmp(&other.minor) {
            Ordering::Equal => {}
            ord => return ord,
        }
        match self.patch.cmp(&other.patch) {
            Ordering::Equal => {}
            ord => return ord,
        }

        // Prerelease versions have lower precedence
        match (&self.prerelease, &other.prerelease) {
            (None, None) => Ordering::Equal,
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (Some(a), Some(b)) => a.cmp(b),
        }
    }
}

impl PartialOrd for Version {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Version requirement for dependencies
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VersionReq {
    /// Exact version match
    Exact(Version),
    /// Greater than or equal
    Gte(Version),
    /// Less than
    Lt(Version),
    /// Compatible with (^)
    Compatible(Version),
    /// Tilde requirement (~)
    Tilde(Version),
    /// Any version
    Any,
}

impl VersionReq {
    /// Parse version requirement from string
    pub fn parse(s: &str) -> AppStoreResult<Self> {
        let s = s.trim();

        if s == "*" {
            return Ok(VersionReq::Any);
        }

        if s.starts_with("^") {
            return Ok(VersionReq::Compatible(Version::parse(&s[1..])?));
        }

        if s.starts_with("~") {
            return Ok(VersionReq::Tilde(Version::parse(&s[1..])?));
        }

        if s.starts_with(">=") {
            return Ok(VersionReq::Gte(Version::parse(&s[2..])?));
        }

        if s.starts_with("<") {
            return Ok(VersionReq::Lt(Version::parse(&s[1..])?));
        }

        if s.starts_with("=") {
            return Ok(VersionReq::Exact(Version::parse(&s[1..])?));
        }

        // Default to exact match
        Ok(VersionReq::Exact(Version::parse(s)?))
    }

    /// Check if a version satisfies this requirement
    pub fn matches(&self, version: &Version) -> bool {
        match self {
            VersionReq::Exact(v) => version == v,
            VersionReq::Gte(v) => version >= v,
            VersionReq::Lt(v) => version < v,
            VersionReq::Compatible(v) => version.is_compatible_with(v) && version >= v,
            VersionReq::Tilde(v) => {
                version.major == v.major
                    && version.minor == v.minor
                    && version.patch >= v.patch
            }
            VersionReq::Any => true,
        }
    }
}

impl fmt::Display for VersionReq {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VersionReq::Exact(v) => write!(f, "={}", v),
            VersionReq::Gte(v) => write!(f, ">={}", v),
            VersionReq::Lt(v) => write!(f, "<{}", v),
            VersionReq::Compatible(v) => write!(f, "^{}", v),
            VersionReq::Tilde(v) => write!(f, "~{}", v),
            VersionReq::Any => write!(f, "*"),
        }
    }
}

impl Default for VersionReq {
    fn default() -> Self {
        VersionReq::Any
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_parse() {
        let v = Version::parse("1.2.3").unwrap();
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);

        let v = Version::parse("v2.0.0-alpha").unwrap();
        assert_eq!(v.major, 2);
        assert_eq!(v.prerelease, Some("alpha".to_string()));

        let v = Version::parse("1.0.0+build.123").unwrap();
        assert_eq!(v.build, Some("build.123".to_string()));
    }

    #[test]
    fn test_version_compare() {
        assert!(Version::parse("1.2.3").unwrap() < Version::parse("2.0.0").unwrap());
        assert!(Version::parse("1.2.3").unwrap() > Version::parse("1.2.2").unwrap());
        assert!(Version::parse("1.0.0-alpha").unwrap() < Version::parse("1.0.0").unwrap());
    }

    #[test]
    fn test_version_bump() {
        let v = Version::new(1, 2, 3);
        assert_eq!(v.bump_patch(), Version::new(1, 2, 4));
        assert_eq!(v.bump_minor(), Version::new(1, 3, 0));
        assert_eq!(v.bump_major(), Version::new(2, 0, 0));
    }

    #[test]
    fn test_version_requirement() {
        let v100 = Version::parse("1.0.0").unwrap();
        let v110 = Version::parse("1.1.0").unwrap();
        let v200 = Version::parse("2.0.0").unwrap();

        let req = VersionReq::parse("^1.0.0").unwrap();
        assert!(req.matches(&v100));
        assert!(req.matches(&v110));
        assert!(!req.matches(&v200));

        let req = VersionReq::parse("~1.0.0").unwrap();
        assert!(req.matches(&v100));
        assert!(!req.matches(&v110)); // Tilde only allows patch updates
    }

    #[test]
    fn test_version_compatibility() {
        let v1 = Version::parse("1.2.3").unwrap();
        let v2 = Version::parse("1.5.0").unwrap();
        let v3 = Version::parse("2.0.0").unwrap();

        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }
}
