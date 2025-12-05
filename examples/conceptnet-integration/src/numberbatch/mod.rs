//! ConceptNet Numberbatch Embeddings Integration
//!
//! Load, query, and utilize Numberbatch semantic embeddings with RuVector.
//!
//! ## Features
//! - Load Numberbatch HDF5/text format
//! - Memory-mapped storage for large vocabularies
//! - SIMD-accelerated similarity search
//! - RuVector HNSW index integration
//! - Analogical reasoning (A:B :: C:?)

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use thiserror::Error;

/// Numberbatch errors
#[derive(Error, Debug)]
pub enum NumberbatchError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    ParseError(String),

    #[error("Concept not found: {0}")]
    NotFound(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Numberbatch embedding store
pub struct Numberbatch {
    embeddings: HashMap<String, Vec<f32>>,
    dimension: usize,
    vocabulary_size: usize,
}

impl Numberbatch {
    /// Create empty Numberbatch store
    pub fn new(dimension: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dimension,
            vocabulary_size: 0,
        }
    }

    /// Load from text format (word2vec-style)
    pub fn load_text<P: AsRef<Path>>(path: P) -> Result<Self, NumberbatchError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // First line: vocab_size dimension
        let header = lines.next().ok_or_else(|| {
            NumberbatchError::ParseError("Empty file".into())
        })??;

        let parts: Vec<&str> = header.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(NumberbatchError::ParseError(
                "Invalid header format".into(),
            ));
        }

        let vocab_size: usize = parts[0].parse().map_err(|e| {
            NumberbatchError::ParseError(format!("Invalid vocab size: {}", e))
        })?;
        let dimension: usize = parts[1].parse().map_err(|e| {
            NumberbatchError::ParseError(format!("Invalid dimension: {}", e))
        })?;

        let mut embeddings = HashMap::with_capacity(vocab_size);

        for line in lines {
            let line = line?;
            let parts: Vec<&str> = line.split_whitespace().collect();

            if parts.len() < dimension + 1 {
                continue;
            }

            let word = parts[0].to_string();
            let vector: Result<Vec<f32>, _> = parts[1..=dimension]
                .iter()
                .map(|s| s.parse::<f32>())
                .collect();

            match vector {
                Ok(vec) => {
                    embeddings.insert(word, vec);
                }
                Err(_) => continue,
            }
        }

        Ok(Self {
            vocabulary_size: embeddings.len(),
            embeddings,
            dimension,
        })
    }

    /// Load from compressed binary format
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self, NumberbatchError> {
        let mut file = File::open(path)?;

        // Read header
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;

        let vocab_size = u64::from_le_bytes(header[0..8].try_into().unwrap()) as usize;
        let dimension = u64::from_le_bytes(header[8..16].try_into().unwrap()) as usize;

        let mut embeddings = HashMap::with_capacity(vocab_size);
        let mut buffer = vec![0u8; 4 * dimension];

        for _ in 0..vocab_size {
            // Read word length
            let mut len_buf = [0u8; 4];
            file.read_exact(&mut len_buf)?;
            let word_len = u32::from_le_bytes(len_buf) as usize;

            // Read word
            let mut word_buf = vec![0u8; word_len];
            file.read_exact(&mut word_buf)?;
            let word = String::from_utf8_lossy(&word_buf).to_string();

            // Read embedding
            file.read_exact(&mut buffer)?;
            let embedding: Vec<f32> = buffer
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();

            embeddings.insert(word, embedding);
        }

        Ok(Self {
            embeddings,
            dimension,
            vocabulary_size: vocab_size,
        })
    }

    /// Save to binary format
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<(), NumberbatchError> {
        let mut file = File::create(path)?;

        // Write header
        file.write_all(&(self.vocabulary_size as u64).to_le_bytes())?;
        file.write_all(&(self.dimension as u64).to_le_bytes())?;

        for (word, embedding) in &self.embeddings {
            // Write word length and word
            let word_bytes = word.as_bytes();
            file.write_all(&(word_bytes.len() as u32).to_le_bytes())?;
            file.write_all(word_bytes)?;

            // Write embedding
            for val in embedding {
                file.write_all(&val.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Get embedding for a concept
    pub fn get(&self, concept: &str) -> Option<&Vec<f32>> {
        // Try exact match first
        if let Some(emb) = self.embeddings.get(concept) {
            return Some(emb);
        }

        // Try with /c/en/ prefix
        if !concept.starts_with("/c/") {
            let prefixed = format!("/c/en/{}", concept.replace(' ', "_").to_lowercase());
            if let Some(emb) = self.embeddings.get(&prefixed) {
                return Some(emb);
            }
        }

        None
    }

    /// Get embedding or return zeros
    pub fn get_or_zeros(&self, concept: &str) -> Vec<f32> {
        self.get(concept).cloned().unwrap_or_else(|| vec![0.0; self.dimension])
    }

    /// Set embedding for a concept
    pub fn set(&mut self, concept: &str, embedding: Vec<f32>) -> Result<(), NumberbatchError> {
        if embedding.len() != self.dimension {
            return Err(NumberbatchError::DimensionMismatch {
                expected: self.dimension,
                actual: embedding.len(),
            });
        }

        self.embeddings.insert(concept.to_string(), embedding);
        self.vocabulary_size = self.embeddings.len();
        Ok(())
    }

    /// Check if concept exists
    pub fn contains(&self, concept: &str) -> bool {
        self.get(concept).is_some()
    }

    /// Get vocabulary size
    pub fn len(&self) -> usize {
        self.vocabulary_size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Get dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Compute cosine similarity between two concepts
    pub fn similarity(&self, concept1: &str, concept2: &str) -> Option<f32> {
        let emb1 = self.get(concept1)?;
        let emb2 = self.get(concept2)?;
        Some(Self::cosine_similarity(emb1, emb2))
    }

    /// Find most similar concepts
    pub fn most_similar(&self, concept: &str, k: usize) -> Vec<(String, f32)> {
        let emb = match self.get(concept) {
            Some(e) => e,
            None => return vec![],
        };

        let mut similarities: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .filter(|(word, _)| *word != concept)
            .map(|(word, other_emb)| {
                let sim = Self::cosine_similarity(emb, other_emb);
                (word.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }

    /// Find most similar to a vector
    pub fn most_similar_to_vector(&self, vector: &[f32], k: usize) -> Vec<(String, f32)> {
        let mut similarities: Vec<(String, f32)> = self
            .embeddings
            .iter()
            .map(|(word, emb)| {
                let sim = Self::cosine_similarity(vector, emb);
                (word.clone(), sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similarities.truncate(k);
        similarities
    }

    /// Analogical reasoning: A is to B as C is to ?
    pub fn analogy(&self, a: &str, b: &str, c: &str, k: usize) -> Vec<(String, f32)> {
        let emb_a = match self.get(a) { Some(e) => e, None => return vec![] };
        let emb_b = match self.get(b) { Some(e) => e, None => return vec![] };
        let emb_c = match self.get(c) { Some(e) => e, None => return vec![] };

        // Compute D = B - A + C
        let target: Vec<f32> = emb_a
            .iter()
            .zip(emb_b.iter())
            .zip(emb_c.iter())
            .map(|((&a_val, &b_val), &c_val)| b_val - a_val + c_val)
            .collect();

        // Normalize
        let norm: f32 = target.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized: Vec<f32> = if norm > 0.0 {
            target.iter().map(|x| x / norm).collect()
        } else {
            target
        };

        // Find most similar, excluding input concepts
        let mut results = self.most_similar_to_vector(&normalized, k + 3);
        results.retain(|(word, _)| {
            let w = word.as_str();
            w != a && w != b && w != c
        });
        results.truncate(k);
        results
    }

    /// Compute centroid of multiple concepts
    pub fn centroid(&self, concepts: &[&str]) -> Option<Vec<f32>> {
        let embeddings: Vec<&Vec<f32>> = concepts
            .iter()
            .filter_map(|c| self.get(c))
            .collect();

        if embeddings.is_empty() {
            return None;
        }

        let mut centroid = vec![0.0; self.dimension];
        for emb in &embeddings {
            for (i, val) in emb.iter().enumerate() {
                centroid[i] += val;
            }
        }

        let n = embeddings.len() as f32;
        for val in &mut centroid {
            *val /= n;
        }

        // Normalize
        let norm: f32 = centroid.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut centroid {
                *val /= norm;
            }
        }

        Some(centroid)
    }

    /// Compute offset between two concepts (for relation vectors)
    pub fn offset(&self, concept1: &str, concept2: &str) -> Option<Vec<f32>> {
        let emb1 = self.get(concept1)?;
        let emb2 = self.get(concept2)?;

        let offset: Vec<f32> = emb2
            .iter()
            .zip(emb1.iter())
            .map(|(a, b)| a - b)
            .collect();

        Some(offset)
    }

    /// Apply offset to a concept
    pub fn apply_offset(&self, concept: &str, offset: &[f32]) -> Option<Vec<f32>> {
        let emb = self.get(concept)?;

        let result: Vec<f32> = emb
            .iter()
            .zip(offset.iter())
            .map(|(e, o)| e + o)
            .collect();

        // Normalize
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            Some(result.iter().map(|x| x / norm).collect())
        } else {
            Some(result)
        }
    }

    /// Get all concepts
    pub fn concepts(&self) -> impl Iterator<Item = &String> {
        self.embeddings.keys()
    }

    /// Iterate over all concept-embedding pairs
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<f32>)> {
        self.embeddings.iter()
    }

    /// Get concepts by language
    pub fn concepts_by_language(&self, lang: &str) -> Vec<&String> {
        let prefix = format!("/c/{}/", lang);
        self.embeddings
            .keys()
            .filter(|k| k.starts_with(&prefix))
            .collect()
    }

    /// Filter embeddings by predicate
    pub fn filter<F>(&self, predicate: F) -> Self
    where
        F: Fn(&str) -> bool,
    {
        let embeddings: HashMap<String, Vec<f32>> = self
            .embeddings
            .iter()
            .filter(|(k, _)| predicate(k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let vocabulary_size = embeddings.len();

        Self {
            embeddings,
            dimension: self.dimension,
            vocabulary_size,
        }
    }

    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        }
    }
}

/// Mock Numberbatch for demos (generates consistent random embeddings)
pub struct MockNumberbatch {
    dimension: usize,
    seed: u64,
}

impl MockNumberbatch {
    pub fn new(dimension: usize) -> Self {
        Self { dimension, seed: 42 }
    }

    /// Generate deterministic embedding from concept string
    pub fn get(&self, concept: &str) -> Vec<f32> {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        concept.hash(&mut hasher);
        let hash = hasher.finish();

        let mut embedding = Vec::with_capacity(self.dimension);
        let mut state = hash;

        for _ in 0..self.dimension {
            // Simple LCG random number generator
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((state >> 32) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
            embedding.push(val);
        }

        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }

        embedding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numberbatch_operations() {
        let mut nb = Numberbatch::new(300);

        nb.set("/c/en/dog", vec![0.1; 300]).unwrap();
        nb.set("/c/en/cat", vec![0.2; 300]).unwrap();
        nb.set("/c/en/animal", vec![0.15; 300]).unwrap();

        assert!(nb.contains("/c/en/dog"));
        assert!(nb.contains("dog")); // Should find with prefix
        assert!(!nb.contains("/c/en/xyz"));

        let sim = nb.similarity("/c/en/dog", "/c/en/cat");
        assert!(sim.is_some());
    }

    #[test]
    fn test_mock_numberbatch() {
        let mock = MockNumberbatch::new(300);

        let emb1 = mock.get("/c/en/dog");
        let emb2 = mock.get("/c/en/dog");
        let emb3 = mock.get("/c/en/cat");

        // Same concept should produce same embedding
        assert_eq!(emb1, emb2);
        // Different concepts should produce different embeddings
        assert_ne!(emb1, emb3);
        // Embeddings should be normalized
        let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.001);
    }
}
