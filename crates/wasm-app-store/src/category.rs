//! App categories and classification

use serde::{Deserialize, Serialize};

/// App category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AppCategory {
    /// Utility and tool apps
    Utilities,
    /// Data processing and transformation
    DataProcessing,
    /// AI and machine learning
    AiMl,
    /// Cryptography and security
    Crypto,
    /// Image and media processing
    Media,
    /// Text and document processing
    Text,
    /// Math and scientific computing
    Math,
    /// Games and entertainment
    Games,
    /// Financial and business
    Finance,
    /// Development tools
    DevTools,
    /// Web and API integration
    Web,
    /// IoT and embedded
    Embedded,
    /// Education and learning
    Education,
    /// Social and communication
    Social,
    /// Health and fitness
    Health,
    /// Other/uncategorized
    Other,
}

impl Default for AppCategory {
    fn default() -> Self {
        AppCategory::Other
    }
}

impl AppCategory {
    /// Get all categories
    pub fn all() -> Vec<AppCategory> {
        vec![
            AppCategory::Utilities,
            AppCategory::DataProcessing,
            AppCategory::AiMl,
            AppCategory::Crypto,
            AppCategory::Media,
            AppCategory::Text,
            AppCategory::Math,
            AppCategory::Games,
            AppCategory::Finance,
            AppCategory::DevTools,
            AppCategory::Web,
            AppCategory::Embedded,
            AppCategory::Education,
            AppCategory::Social,
            AppCategory::Health,
            AppCategory::Other,
        ]
    }

    /// Get category display name
    pub fn display_name(&self) -> &'static str {
        match self {
            AppCategory::Utilities => "Utilities",
            AppCategory::DataProcessing => "Data Processing",
            AppCategory::AiMl => "AI & Machine Learning",
            AppCategory::Crypto => "Cryptography & Security",
            AppCategory::Media => "Media Processing",
            AppCategory::Text => "Text Processing",
            AppCategory::Math => "Math & Science",
            AppCategory::Games => "Games",
            AppCategory::Finance => "Finance",
            AppCategory::DevTools => "Developer Tools",
            AppCategory::Web => "Web & APIs",
            AppCategory::Embedded => "Embedded & IoT",
            AppCategory::Education => "Education",
            AppCategory::Social => "Social",
            AppCategory::Health => "Health & Fitness",
            AppCategory::Other => "Other",
        }
    }

    /// Get category icon (emoji)
    pub fn icon(&self) -> &'static str {
        match self {
            AppCategory::Utilities => "🔧",
            AppCategory::DataProcessing => "📊",
            AppCategory::AiMl => "🤖",
            AppCategory::Crypto => "🔐",
            AppCategory::Media => "🎬",
            AppCategory::Text => "📝",
            AppCategory::Math => "🔢",
            AppCategory::Games => "🎮",
            AppCategory::Finance => "💰",
            AppCategory::DevTools => "👨‍💻",
            AppCategory::Web => "🌐",
            AppCategory::Embedded => "📟",
            AppCategory::Education => "📚",
            AppCategory::Social => "💬",
            AppCategory::Health => "❤️",
            AppCategory::Other => "📦",
        }
    }
}

/// App platform target
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Platform {
    /// Web browser
    Browser,
    /// Node.js
    NodeJs,
    /// Deno
    Deno,
    /// Cloudflare Workers
    CloudflareWorkers,
    /// Embedded/IoT
    Embedded,
    /// Mobile (React Native, etc.)
    Mobile,
    /// Desktop (Electron, Tauri)
    Desktop,
    /// Universal (all platforms)
    Universal,
}

impl Default for Platform {
    fn default() -> Self {
        Platform::Universal
    }
}

impl Platform {
    /// Check if this platform supports WASI
    pub fn supports_wasi(&self) -> bool {
        matches!(
            self,
            Platform::NodeJs | Platform::Deno | Platform::CloudflareWorkers | Platform::Universal
        )
    }

    /// Check if this platform runs in browser
    pub fn is_browser(&self) -> bool {
        matches!(self, Platform::Browser | Platform::Universal)
    }
}

/// App content rating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentRating {
    /// Everyone
    Everyone,
    /// Teen (13+)
    Teen,
    /// Mature (17+)
    Mature,
}

impl Default for ContentRating {
    fn default() -> Self {
        ContentRating::Everyone
    }
}

/// App license type
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum License {
    /// MIT License
    Mit,
    /// Apache 2.0
    Apache2,
    /// GPL v3
    Gpl3,
    /// BSD 3-Clause
    Bsd3,
    /// Proprietary
    Proprietary,
    /// Custom license
    Custom(String),
}

impl Default for License {
    fn default() -> Self {
        License::Mit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_categories() {
        let categories = AppCategory::all();
        assert!(!categories.is_empty());

        for cat in categories {
            assert!(!cat.display_name().is_empty());
            assert!(!cat.icon().is_empty());
        }
    }

    #[test]
    fn test_platform() {
        assert!(Platform::NodeJs.supports_wasi());
        assert!(Platform::Browser.is_browser());
        assert!(Platform::Universal.is_browser());
        assert!(Platform::Universal.supports_wasi());
    }
}
