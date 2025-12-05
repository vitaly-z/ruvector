//! Numberbatch Embeddings Loader
//!
//! Loads the full Numberbatch semantic embeddings (~400K concepts, 300-dim).
//! Supports multiple formats:
//! - Text format (.txt, .txt.gz) - one line per concept
//! - HDF5 format (.h5) - for efficient random access
//! - Binary format for fastest loading
//!
//! ## Data Source
//! https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz

use crate::numberbatch::Numberbatch;
use flate2::read::GzDecoder;
use memmap2::Mmap;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;

/// Configuration for Numberbatch loading
#[derive(Debug, Clone)]
pub struct NumberbatchConfig {
    /// Path to embeddings file
    pub file_path: String,
    /// Languages to load (empty = all)
    pub languages: Vec<String>,
    /// Load only concepts present in a given list
    pub concept_filter: Option<Vec<String>>,
    /// Normalize vectors to unit length
    pub normalize: bool,
    /// Use memory-mapped loading for large files
    pub use_mmap: bool,
    /// Number of parallel workers for parsing
    pub num_workers: usize,
}

impl Default for NumberbatchConfig {
    fn default() -> Self {
        Self {
            file_path: String::new(),
            languages: vec!["en".to_string()],
            concept_filter: None,
            normalize: true,
            use_mmap: true,
            num_workers: num_cpus::get(),
        }
    }
}

/// Numberbatch embedding loader
pub struct NumberbatchLoader {
    config: NumberbatchConfig,
}

impl NumberbatchLoader {
    /// Create a new loader
    pub fn new(config: NumberbatchConfig) -> Self {
        Self { config }
    }

    /// Load embeddings into a Numberbatch instance
    pub fn load(&self) -> Result<Numberbatch, LoadError> {
        let path = Path::new(&self.config.file_path);

        if !path.exists() {
            return Err(LoadError::FileNotFound(self.config.file_path.clone()));
        }

        // Detect format
        if self.config.file_path.ends_with(".bin") {
            self.load_binary()
        } else if self.config.file_path.ends_with(".h5") {
            self.load_hdf5()
        } else {
            self.load_text()
        }
    }

    /// Load from text format (word2vec style)
    fn load_text(&self) -> Result<Numberbatch, LoadError> {
        let file = File::open(&self.config.file_path)?;
        let reader: Box<dyn BufRead> = if self.config.file_path.ends_with(".gz") {
            Box::new(BufReader::with_capacity(
                8 * 1024 * 1024,
                GzDecoder::new(file),
            ))
        } else {
            Box::new(BufReader::with_capacity(8 * 1024 * 1024, file))
        };

        let languages: std::collections::HashSet<&str> =
            self.config.languages.iter().map(|s| s.as_str()).collect();
        let filter: Option<std::collections::HashSet<&str>> = self.config.concept_filter.as_ref().map(|v| {
            v.iter().map(|s| s.as_str()).collect()
        });

        let mut lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

        // First line might be header (vocab_size dim)
        let (vocab_size, dim) = if let Some(first) = lines.first() {
            let parts: Vec<&str> = first.split_whitespace().collect();
            if parts.len() == 2 {
                if let (Ok(v), Ok(d)) = (parts[0].parse::<usize>(), parts[1].parse::<usize>()) {
                    lines.remove(0);
                    (v, d)
                } else {
                    (0, 300) // Default dimension
                }
            } else {
                (0, 300)
            }
        } else {
            return Err(LoadError::Parse("Empty file".to_string()));
        };

        println!(
            "Loading Numberbatch: {} concepts, {}-dimensional",
            if vocab_size > 0 {
                vocab_size.to_string()
            } else {
                "unknown".to_string()
            },
            dim
        );

        // Parse in parallel
        let embeddings: Vec<Option<(String, Vec<f32>)>> = lines
            .par_iter()
            .map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() < 2 {
                    return None;
                }

                let concept = parts[0];

                // Language filter
                if !languages.is_empty() {
                    let lang = Self::extract_language(concept);
                    if !languages.contains(lang) {
                        return None;
                    }
                }

                // Concept filter
                if let Some(ref filter_set) = filter {
                    if !filter_set.contains(concept) {
                        return None;
                    }
                }

                // Parse vector
                let mut vec: Vec<f32> = parts[1..]
                    .iter()
                    .filter_map(|s| s.parse::<f32>().ok())
                    .collect();

                if vec.is_empty() {
                    return None;
                }

                // Normalize if requested
                if self.config.normalize {
                    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for v in &mut vec {
                            *v /= norm;
                        }
                    }
                }

                Some((concept.to_string(), vec))
            })
            .collect();

        // Build Numberbatch
        let mut nb = Numberbatch::new(dim);
        let mut count = 0;

        for emb_opt in embeddings {
            if let Some((concept, vec)) = emb_opt {
                if nb.set(&concept, vec).is_ok() {
                    count += 1;
                }
            }
        }

        println!("Loaded {} embeddings", count);
        Ok(nb)
    }

    /// Load from binary format (faster)
    fn load_binary(&self) -> Result<Numberbatch, LoadError> {
        let file = File::open(&self.config.file_path)?;

        if self.config.use_mmap {
            // Memory-mapped loading
            let mmap = unsafe { Mmap::map(&file)? };

            // Binary format: [dim: u32] [count: u32] [entries...]
            // Entry: [concept_len: u16] [concept: bytes] [vector: f32 * dim]
            if mmap.len() < 8 {
                return Err(LoadError::Parse("Invalid binary file".to_string()));
            }

            let dim = u32::from_le_bytes([mmap[0], mmap[1], mmap[2], mmap[3]]) as usize;
            let count = u32::from_le_bytes([mmap[4], mmap[5], mmap[6], mmap[7]]) as usize;

            let mut nb = Numberbatch::new(dim);
            let mut offset = 8;

            for _ in 0..count {
                if offset + 2 > mmap.len() {
                    break;
                }

                let concept_len = u16::from_le_bytes([mmap[offset], mmap[offset + 1]]) as usize;
                offset += 2;

                if offset + concept_len + dim * 4 > mmap.len() {
                    break;
                }

                let concept = String::from_utf8_lossy(&mmap[offset..offset + concept_len]).to_string();
                offset += concept_len;

                let mut vec = Vec::with_capacity(dim);
                for _ in 0..dim {
                    let bytes = [
                        mmap[offset],
                        mmap[offset + 1],
                        mmap[offset + 2],
                        mmap[offset + 3],
                    ];
                    vec.push(f32::from_le_bytes(bytes));
                    offset += 4;
                }

                // Apply filters
                let lang = Self::extract_language(&concept);
                if !self.config.languages.is_empty()
                    && !self.config.languages.iter().any(|l| l == lang)
                {
                    continue;
                }

                let _ = nb.set(&concept, vec);
            }

            Ok(nb)
        } else {
            // Standard file reading
            let mut reader = BufReader::new(file);
            let mut buf = [0u8; 4];

            reader.read_exact(&mut buf)?;
            let dim = u32::from_le_bytes(buf) as usize;

            reader.read_exact(&mut buf)?;
            let count = u32::from_le_bytes(buf) as usize;

            let mut nb = Numberbatch::new(dim);

            for _ in 0..count {
                let mut len_buf = [0u8; 2];
                if reader.read_exact(&mut len_buf).is_err() {
                    break;
                }
                let concept_len = u16::from_le_bytes(len_buf) as usize;

                let mut concept_buf = vec![0u8; concept_len];
                reader.read_exact(&mut concept_buf)?;
                let concept = String::from_utf8_lossy(&concept_buf).to_string();

                let mut vec = Vec::with_capacity(dim);
                for _ in 0..dim {
                    reader.read_exact(&mut buf)?;
                    vec.push(f32::from_le_bytes(buf));
                }

                let _ = nb.set(&concept, vec);
            }

            Ok(nb)
        }
    }

    /// Load from HDF5 format
    fn load_hdf5(&self) -> Result<Numberbatch, LoadError> {
        // HDF5 support would require hdf5 crate
        // For now, return error with helpful message
        Err(LoadError::Parse(
            "HDF5 format not supported. Convert to text or binary format first.\n\
             Use: python -c \"import h5py; f=h5py.File('numberbatch.h5'); ...\"".to_string()
        ))
    }

    /// Save to binary format for faster future loading
    pub fn save_binary(nb: &Numberbatch, path: &str) -> Result<(), LoadError> {
        let mut file = File::create(path)?;

        let dim = nb.dimension() as u32;
        let count = nb.len() as u32;

        file.write_all(&dim.to_le_bytes())?;
        file.write_all(&count.to_le_bytes())?;

        for (concept, vec) in nb.iter() {
            let concept_bytes = concept.as_bytes();
            let concept_len = concept_bytes.len() as u16;

            file.write_all(&concept_len.to_le_bytes())?;
            file.write_all(concept_bytes)?;

            for val in vec {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    fn extract_language(uri: &str) -> &str {
        // /c/en/dog -> en
        let parts: Vec<&str> = uri.split('/').collect();
        if parts.len() >= 3 {
            parts[2]
        } else {
            "en"
        }
    }
}

/// Download Numberbatch embeddings
pub async fn download_numberbatch(output_path: &str, multilingual: bool) -> Result<(), LoadError> {
    let url = if multilingual {
        "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-19.08.txt.gz"
    } else {
        "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz"
    };

    println!("Downloading Numberbatch from: {}", url);
    println!(
        "This is a ~{}MB download...",
        if multilingual { "1200" } else { "150" }
    );

    let response = reqwest::get(url).await.map_err(|e| LoadError::Parse(e.to_string()))?;
    let bytes = response.bytes().await.map_err(|e| LoadError::Parse(e.to_string()))?;

    std::fs::write(output_path, bytes)?;
    println!("Downloaded to: {}", output_path);

    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("File not found: {0}")]
    FileNotFound(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_language() {
        assert_eq!(NumberbatchLoader::extract_language("/c/en/dog"), "en");
        assert_eq!(NumberbatchLoader::extract_language("/c/de/hund"), "de");
    }
}
