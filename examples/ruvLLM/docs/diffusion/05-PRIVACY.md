# Privacy Tiers and Encryption

## Overview

RuvDLLM implements a comprehensive multi-tier privacy system that allows users to control exactly how their learning data is shared. Each tier provides different levels of protection, from fully private (never leaves device) to public (globally aggregated with strong privacy guarantees).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Privacy Tier Architecture                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 0: PRIVATE                                                     │   │
│  │ • E2E encrypted, device-only                                        │   │
│  │ • User personal patterns                                            │   │
│  │ • Never transmitted                                                 │   │
│  │ • Key: Device-bound, non-exportable                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          │ Explicit consent + anonymization                 │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 1: GROUP                                                       │   │
│  │ • Group key encryption                                              │   │
│  │ • Team/project patterns                                             │   │
│  │ • Shared within key holders                                         │   │
│  │ • Key: Distributed via secure key exchange                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          │ Aggregation threshold (k≥5)                      │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 2: TENANT                                                      │   │
│  │ • Organization-wide access                                          │   │
│  │ • Department/company patterns                                       │   │
│  │ • Role-based access control                                         │   │
│  │ • Key: HSM-backed, org-managed                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                          │                                                  │
│                          │ Differential privacy (ε=1.0) + k-anonymity       │
│                          ▼                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ TIER 3: PUBLIC                                                      │   │
│  │ • Global federation                                                 │   │
│  │ • Common patterns                                                   │   │
│  │ • Fully anonymized                                                  │   │
│  │ • Key: Public aggregation keys                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Privacy Tier Implementation

### Core Types

```rust
/// Privacy tiers for pattern sharing
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PrivacyTier {
    /// Never leaves device, E2E encrypted at rest
    Private,
    /// Shared within group key holders
    Group,
    /// Organization-wide with RBAC
    Tenant,
    /// Global federation with DP guarantees
    Public,
}

/// Privacy-aware pattern storage
pub struct PrivatePattern {
    /// Pattern ID (opaque)
    pub id: PatternId,
    /// Privacy tier
    pub tier: PrivacyTier,
    /// Encrypted LoRA weights
    pub encrypted_weights: EncryptedBlob,
    /// Metadata (searchable only at appropriate tier)
    pub metadata: PatternMetadata,
    /// Consent record
    pub consent: ConsentRecord,
    /// Encryption key reference
    pub key_ref: KeyReference,
}

/// Consent tracking for privacy compliance
#[derive(Clone, Debug)]
pub struct ConsentRecord {
    /// When consent was given
    pub timestamp: DateTime<Utc>,
    /// What was consented to
    pub scope: ConsentScope,
    /// How consent was obtained
    pub method: ConsentMethod,
    /// Revocation capability
    pub revocable: bool,
    /// Expiration (if any)
    pub expires: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug)]
pub enum ConsentScope {
    /// Pattern can be used for local learning only
    LocalOnly,
    /// Pattern can be shared within group
    GroupSharing { group_id: GroupId },
    /// Pattern can be aggregated tenant-wide
    TenantAggregation { tenant_id: TenantId },
    /// Pattern can participate in global federation
    GlobalFederation {
        anonymization_required: bool,
        dp_epsilon: f32,
    },
}
```

### Tier Manager

```rust
/// Privacy tier manager
pub struct PrivacyTierManager {
    /// Local encryption keys
    keys: KeyStore,
    /// Consent database
    consents: ConsentDatabase,
    /// Pattern classifier
    classifier: PatternClassifier,
    /// Anonymization pipeline
    anonymizer: PatternAnonymizer,
}

impl PrivacyTierManager {
    /// Store pattern with appropriate privacy tier
    pub async fn store_pattern(
        &mut self,
        pattern: RawPattern,
        requested_tier: PrivacyTier,
    ) -> Result<PatternId, PrivacyError> {
        // Classify pattern to determine if tier is appropriate
        let classification = self.classifier.classify(&pattern)?;

        // Check if pattern contains PII or sensitive data
        if classification.contains_pii && requested_tier != PrivacyTier::Private {
            return Err(PrivacyError::PIIDetected {
                suggestion: PrivacyTier::Private,
            });
        }

        // Verify consent for requested tier
        let consent = self.get_or_request_consent(requested_tier).await?;

        // Get encryption key for tier
        let key = self.keys.get_key_for_tier(requested_tier)?;

        // Anonymize if needed for higher tiers
        let processed_pattern = if requested_tier >= PrivacyTier::Tenant {
            self.anonymizer.anonymize(&pattern, &classification)?
        } else {
            pattern
        };

        // Encrypt
        let encrypted = self.encrypt_pattern(&processed_pattern, &key)?;

        // Store
        let id = self.storage.store(PrivatePattern {
            id: PatternId::generate(),
            tier: requested_tier,
            encrypted_weights: encrypted,
            metadata: self.create_metadata(&processed_pattern, requested_tier),
            consent,
            key_ref: key.reference(),
        }).await?;

        Ok(id)
    }

    /// Promote pattern to higher tier (requires consent)
    pub async fn promote_pattern(
        &mut self,
        pattern_id: PatternId,
        target_tier: PrivacyTier,
    ) -> Result<(), PrivacyError> {
        let pattern = self.storage.get(pattern_id).await?;

        // Can only promote up
        if target_tier <= pattern.tier {
            return Err(PrivacyError::InvalidPromotion);
        }

        // Get new consent
        let consent = self.request_consent_for_promotion(
            pattern.tier,
            target_tier,
        ).await?;

        // Re-anonymize if needed
        let decrypted = self.decrypt_pattern(&pattern)?;
        let anonymized = if target_tier >= PrivacyTier::Tenant {
            self.anonymizer.anonymize(&decrypted, &pattern.classification)?
        } else {
            decrypted
        };

        // Re-encrypt with new tier's key
        let new_key = self.keys.get_key_for_tier(target_tier)?;
        let encrypted = self.encrypt_pattern(&anonymized, &new_key)?;

        // Update storage
        self.storage.update(pattern_id, PrivatePattern {
            tier: target_tier,
            encrypted_weights: encrypted,
            consent,
            key_ref: new_key.reference(),
            ..pattern
        }).await?;

        Ok(())
    }
}
```

## Encryption Systems

### Key Hierarchy

```rust
/// Hierarchical key management
pub struct KeyStore {
    /// Root key (device-bound, TPM/Secure Enclave)
    root_key: RootKey,
    /// Tier keys derived from root
    tier_keys: HashMap<PrivacyTier, TierKey>,
    /// Group keys (received from key exchange)
    group_keys: HashMap<GroupId, GroupKey>,
    /// Tenant keys (HSM-backed)
    tenant_keys: HashMap<TenantId, TenantKey>,
}

impl KeyStore {
    /// Initialize key hierarchy
    pub fn init(hardware_key: HardwareKey) -> Result<Self, KeyError> {
        // Derive root key from hardware
        let root_key = RootKey::derive_from_hardware(hardware_key)?;

        // Derive tier keys
        let mut tier_keys = HashMap::new();

        // Private tier: Direct derivation, never exportable
        tier_keys.insert(
            PrivacyTier::Private,
            TierKey::derive(&root_key, "private", KeyFlags::NON_EXPORTABLE)?,
        );

        Ok(Self {
            root_key,
            tier_keys,
            group_keys: HashMap::new(),
            tenant_keys: HashMap::new(),
        })
    }

    /// Get key for tier
    pub fn get_key_for_tier(&self, tier: PrivacyTier) -> Result<&dyn EncryptionKey, KeyError> {
        match tier {
            PrivacyTier::Private => {
                self.tier_keys.get(&tier).ok_or(KeyError::NotFound)
            }
            PrivacyTier::Group => {
                // Group key must be explicitly added via key exchange
                Err(KeyError::RequiresGroupId)
            }
            PrivacyTier::Tenant => {
                Err(KeyError::RequiresTenantId)
            }
            PrivacyTier::Public => {
                // Public tier uses aggregation keys, not encryption
                Ok(&PUBLIC_AGGREGATION_KEY)
            }
        }
    }
}

/// Encryption key trait
pub trait EncryptionKey {
    fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedBlob, CryptoError>;
    fn decrypt(&self, ciphertext: &EncryptedBlob) -> Result<Vec<u8>, CryptoError>;
    fn reference(&self) -> KeyReference;
}

/// Private tier key (AES-256-GCM)
pub struct TierKey {
    key_material: Zeroizing<[u8; 32]>,
    flags: KeyFlags,
}

impl EncryptionKey for TierKey {
    fn encrypt(&self, plaintext: &[u8]) -> Result<EncryptedBlob, CryptoError> {
        let nonce = generate_random_nonce();
        let cipher = Aes256Gcm::new_from_slice(&*self.key_material)?;
        let ciphertext = cipher.encrypt(&nonce.into(), plaintext)?;

        Ok(EncryptedBlob {
            algorithm: Algorithm::Aes256Gcm,
            nonce: nonce.to_vec(),
            ciphertext,
            auth_tag: None, // Included in ciphertext for GCM
        })
    }

    fn decrypt(&self, blob: &EncryptedBlob) -> Result<Vec<u8>, CryptoError> {
        let cipher = Aes256Gcm::new_from_slice(&*self.key_material)?;
        let nonce = GenericArray::from_slice(&blob.nonce);
        cipher.decrypt(nonce, blob.ciphertext.as_ref())
            .map_err(|_| CryptoError::DecryptionFailed)
    }
}
```

### Group Key Exchange

```rust
/// Secure group key exchange (X3DH + Double Ratchet inspired)
pub struct GroupKeyExchange {
    /// Local identity key pair
    identity_key: IdentityKeyPair,
    /// Ephemeral key pairs
    ephemeral_keys: VecDeque<EphemeralKeyPair>,
    /// Group memberships
    groups: HashMap<GroupId, GroupMembership>,
}

impl GroupKeyExchange {
    /// Join a group (receive group key)
    pub async fn join_group(
        &mut self,
        group_id: GroupId,
        invitation: GroupInvitation,
    ) -> Result<GroupKey, KeyExchangeError> {
        // Verify invitation signature
        invitation.verify(&self.known_identities)?;

        // Generate ephemeral key pair for this exchange
        let ephemeral = EphemeralKeyPair::generate();

        // X3DH key agreement
        let shared_secret = self.x3dh_receive(
            &invitation.sender_identity,
            &invitation.ephemeral_public,
            &ephemeral,
        )?;

        // Derive group key from shared secret
        let group_key = GroupKey::derive_from_shared_secret(
            &shared_secret,
            &group_id,
            invitation.key_epoch,
        )?;

        // Store membership
        self.groups.insert(group_id, GroupMembership {
            key: group_key.clone(),
            epoch: invitation.key_epoch,
            members: invitation.members.clone(),
            role: GroupRole::Member,
        });

        Ok(group_key)
    }

    /// Create new group (as admin)
    pub async fn create_group(
        &mut self,
        name: String,
        initial_members: Vec<IdentityPublic>,
    ) -> Result<GroupId, KeyExchangeError> {
        let group_id = GroupId::generate();

        // Generate initial group key
        let group_key = GroupKey::generate_random();

        // Create invitations for each member
        for member in &initial_members {
            let invitation = self.create_invitation(
                &group_id,
                member,
                &group_key,
            ).await?;

            // Send invitation via secure channel
            self.send_invitation(member, invitation).await?;
        }

        // Store as admin
        self.groups.insert(group_id, GroupMembership {
            key: group_key,
            epoch: 0,
            members: initial_members,
            role: GroupRole::Admin,
        });

        Ok(group_id)
    }

    /// Rotate group key (admin only)
    pub async fn rotate_key(&mut self, group_id: GroupId) -> Result<(), KeyExchangeError> {
        let membership = self.groups.get_mut(&group_id)
            .ok_or(KeyExchangeError::NotMember)?;

        if membership.role != GroupRole::Admin {
            return Err(KeyExchangeError::NotAdmin);
        }

        // Generate new key
        let new_key = GroupKey::generate_random();
        let new_epoch = membership.epoch + 1;

        // Distribute to all members
        for member in &membership.members {
            let invitation = self.create_key_rotation(
                &group_id,
                member,
                &new_key,
                new_epoch,
            ).await?;

            self.send_invitation(member, invitation).await?;
        }

        membership.key = new_key;
        membership.epoch = new_epoch;

        Ok(())
    }
}
```

## Anonymization Pipeline

### Pattern Anonymization

```rust
/// Pattern anonymization for higher privacy tiers
pub struct PatternAnonymizer {
    /// PII detector
    pii_detector: PIIDetector,
    /// Generalization rules
    generalizer: Generalizer,
    /// Noise injector
    noise_injector: NoiseInjector,
}

impl PatternAnonymizer {
    /// Anonymize pattern for tenant/public sharing
    pub fn anonymize(
        &self,
        pattern: &RawPattern,
        classification: &Classification,
    ) -> Result<AnonymizedPattern, AnonymizationError> {
        let mut result = pattern.clone();

        // Step 1: Remove/redact PII from metadata
        result.metadata = self.pii_detector.redact_metadata(&pattern.metadata)?;

        // Step 2: Generalize specific values
        result.trigger_pattern = self.generalizer.generalize(&pattern.trigger_pattern)?;

        // Step 3: Add noise to weights (local DP)
        result.weights = self.noise_injector.add_noise(
            &pattern.weights,
            classification.sensitivity,
        )?;

        // Step 4: Remove temporal patterns that could identify
        result.timestamps = vec![]; // Remove timing information

        // Step 5: Verify k-anonymity
        if !self.verify_k_anonymity(&result, K_ANONYMITY_THRESHOLD) {
            // Need more generalization
            result.trigger_pattern = self.generalizer.generalize_aggressive(&result.trigger_pattern)?;
        }

        Ok(AnonymizedPattern {
            weights: result.weights,
            metadata: result.metadata,
            trigger_hash: hash_trigger(&result.trigger_pattern), // Hash instead of raw
            anonymization_level: AnonymizationLevel::Full,
        })
    }

    /// Verify pattern meets k-anonymity requirement
    fn verify_k_anonymity(&self, pattern: &RawPattern, k: usize) -> bool {
        // Check if pattern is indistinguishable from at least k-1 others
        let equivalence_class_size = self.find_equivalence_class_size(pattern);
        equivalence_class_size >= k
    }
}

/// PII detection and redaction
pub struct PIIDetector {
    /// Regex patterns for PII
    patterns: Vec<(PIIType, Regex)>,
    /// ML-based detector for complex cases
    ml_detector: Option<PIIMLModel>,
}

impl PIIDetector {
    pub fn detect(&self, text: &str) -> Vec<PIIMatch> {
        let mut matches = Vec::new();

        // Regex-based detection
        for (pii_type, pattern) in &self.patterns {
            for m in pattern.find_iter(text) {
                matches.push(PIIMatch {
                    pii_type: *pii_type,
                    start: m.start(),
                    end: m.end(),
                    confidence: 1.0,
                });
            }
        }

        // ML-based detection for complex patterns
        if let Some(ml) = &self.ml_detector {
            matches.extend(ml.detect(text));
        }

        // Deduplicate overlapping matches
        self.deduplicate(matches)
    }

    pub fn redact_metadata(&self, metadata: &PatternMetadata) -> Result<PatternMetadata, PIIError> {
        let mut result = metadata.clone();

        // Redact all string fields
        for field in result.string_fields_mut() {
            let matches = self.detect(field);
            for m in matches.iter().rev() {
                field.replace_range(m.start..m.end, "[REDACTED]");
            }
        }

        // Remove user identifiers
        result.user_id = None;
        result.device_id = None;
        result.session_id = None;

        Ok(result)
    }
}

#[derive(Clone, Copy)]
pub enum PIIType {
    Email,
    Phone,
    SSN,
    CreditCard,
    IPAddress,
    Name,
    Address,
    DateOfBirth,
    Custom(u32),
}
```

### Differential Privacy

```rust
/// Differential privacy implementation
pub struct DifferentialPrivacy {
    /// Privacy budget (ε)
    epsilon: f32,
    /// Delta parameter
    delta: f32,
    /// Noise mechanism
    mechanism: NoiseMechanism,
    /// Budget tracker
    budget_tracker: BudgetTracker,
}

impl DifferentialPrivacy {
    /// Add noise for gradient privacy
    pub fn privatize_gradient(
        &mut self,
        gradient: &mut [f32],
        sensitivity: f32,
    ) -> Result<(), PrivacyError> {
        // Check budget
        let cost = self.compute_cost(gradient.len());
        self.budget_tracker.reserve(cost)?;

        match self.mechanism {
            NoiseMechanism::Gaussian => {
                self.add_gaussian_noise(gradient, sensitivity)?;
            }
            NoiseMechanism::Laplace => {
                self.add_laplace_noise(gradient, sensitivity)?;
            }
        }

        // Commit budget
        self.budget_tracker.commit(cost);

        Ok(())
    }

    /// Gaussian mechanism for (ε, δ)-DP
    fn add_gaussian_noise(&self, data: &mut [f32], sensitivity: f32) -> Result<(), PrivacyError> {
        // σ = Δf * √(2 * ln(1.25/δ)) / ε
        let sigma = sensitivity * (2.0 * (1.25 / self.delta).ln()).sqrt() / self.epsilon;

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, sigma).unwrap();

        for x in data.iter_mut() {
            *x += normal.sample(&mut rng);
        }

        Ok(())
    }

    /// Laplace mechanism for ε-DP
    fn add_laplace_noise(&self, data: &mut [f32], sensitivity: f32) -> Result<(), PrivacyError> {
        // b = Δf / ε
        let b = sensitivity / self.epsilon;

        let mut rng = thread_rng();
        let laplace = Laplace::new(0.0, b).unwrap();

        for x in data.iter_mut() {
            *x += laplace.sample(&mut rng);
        }

        Ok(())
    }
}

/// Privacy budget tracking
pub struct BudgetTracker {
    /// Total budget
    total_epsilon: f32,
    /// Spent budget
    spent_epsilon: f32,
    /// Per-epoch budget
    epoch_budget: f32,
    /// Current epoch spending
    epoch_spent: f32,
}

impl BudgetTracker {
    /// Reserve budget (fails if insufficient)
    pub fn reserve(&self, cost: f32) -> Result<(), PrivacyError> {
        if self.spent_epsilon + cost > self.total_epsilon {
            return Err(PrivacyError::BudgetExhausted {
                remaining: self.total_epsilon - self.spent_epsilon,
                requested: cost,
            });
        }

        if self.epoch_spent + cost > self.epoch_budget {
            return Err(PrivacyError::EpochBudgetExhausted);
        }

        Ok(())
    }

    /// Commit spent budget
    pub fn commit(&mut self, cost: f32) {
        self.spent_epsilon += cost;
        self.epoch_spent += cost;
    }

    /// Reset epoch budget (called at epoch boundary)
    pub fn reset_epoch(&mut self) {
        self.epoch_spent = 0.0;
    }
}
```

## Consent Management

### Consent Flow

```rust
/// Consent management system
pub struct ConsentManager {
    /// Stored consents
    consents: ConsentDatabase,
    /// Consent UI integration
    ui: Box<dyn ConsentUI>,
    /// Audit log
    audit_log: AuditLog,
}

impl ConsentManager {
    /// Request consent for privacy tier
    pub async fn request_consent(
        &mut self,
        tier: PrivacyTier,
        purpose: ConsentPurpose,
    ) -> Result<ConsentRecord, ConsentError> {
        // Check for existing valid consent
        if let Some(existing) = self.consents.get_valid_consent(tier, &purpose) {
            return Ok(existing);
        }

        // Prepare consent request
        let request = ConsentRequest {
            tier,
            purpose: purpose.clone(),
            data_usage: self.describe_data_usage(tier),
            retention_period: self.get_retention_period(tier),
            revocation_info: self.get_revocation_info(tier),
        };

        // Show consent UI
        let response = self.ui.show_consent_dialog(request).await?;

        match response {
            ConsentResponse::Granted { scope, duration } => {
                let record = ConsentRecord {
                    timestamp: Utc::now(),
                    scope,
                    method: ConsentMethod::ExplicitUI,
                    revocable: true,
                    expires: duration.map(|d| Utc::now() + d),
                };

                // Store consent
                self.consents.store(tier, &purpose, &record).await?;

                // Audit log
                self.audit_log.log_consent_granted(tier, &purpose, &record);

                Ok(record)
            }
            ConsentResponse::Denied => {
                self.audit_log.log_consent_denied(tier, &purpose);
                Err(ConsentError::Denied)
            }
        }
    }

    /// Revoke consent
    pub async fn revoke_consent(
        &mut self,
        tier: PrivacyTier,
        purpose: &ConsentPurpose,
    ) -> Result<(), ConsentError> {
        // Mark consent as revoked
        self.consents.revoke(tier, purpose).await?;

        // Trigger data deletion/anonymization
        self.handle_revocation(tier, purpose).await?;

        // Audit log
        self.audit_log.log_consent_revoked(tier, purpose);

        Ok(())
    }

    /// Handle consent revocation (delete/anonymize data)
    async fn handle_revocation(
        &mut self,
        tier: PrivacyTier,
        purpose: &ConsentPurpose,
    ) -> Result<(), ConsentError> {
        match tier {
            PrivacyTier::Private => {
                // Nothing to do - data never left device
            }
            PrivacyTier::Group => {
                // Notify group to remove this user's contributions
                self.notify_group_revocation(purpose).await?;
            }
            PrivacyTier::Tenant => {
                // Request tenant-level data deletion
                self.request_tenant_deletion(purpose).await?;
            }
            PrivacyTier::Public => {
                // Data already anonymized and aggregated
                // Record revocation to prevent future contributions
                self.block_future_public_contributions(purpose).await?;
            }
        }

        Ok(())
    }
}

/// Consent purposes
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ConsentPurpose {
    /// Model improvement
    ModelImprovement,
    /// Pattern sharing
    PatternSharing { category: String },
    /// Research
    Research { study_id: String },
    /// Analytics
    Analytics,
    /// Custom purpose
    Custom { id: String, description: String },
}
```

## Data Retention

```rust
/// Data retention policy enforcement
pub struct RetentionManager {
    /// Retention policies per tier
    policies: HashMap<PrivacyTier, RetentionPolicy>,
    /// Cleanup scheduler
    scheduler: RetentionScheduler,
}

#[derive(Clone)]
pub struct RetentionPolicy {
    /// Maximum age for data
    pub max_age: Duration,
    /// Action at expiration
    pub expiration_action: ExpirationAction,
    /// Grace period before deletion
    pub grace_period: Duration,
    /// Whether to notify before deletion
    pub notify_before_deletion: bool,
}

#[derive(Clone)]
pub enum ExpirationAction {
    /// Delete data completely
    Delete,
    /// Anonymize and move to public tier
    AnonymizeAndPromote,
    /// Archive to cold storage
    Archive { storage_tier: StorageTier },
    /// Extend if actively used
    ExtendIfActive { extension: Duration },
}

impl RetentionManager {
    /// Run retention cleanup
    pub async fn run_cleanup(&mut self) -> Result<CleanupReport, RetentionError> {
        let mut report = CleanupReport::default();

        for (tier, policy) in &self.policies {
            let expired = self.find_expired_patterns(*tier, policy).await?;

            for pattern in expired {
                // Notify if required
                if policy.notify_before_deletion {
                    self.notify_expiration(&pattern).await?;
                }

                // Check grace period
                if pattern.grace_period_expired(policy.grace_period) {
                    // Execute expiration action
                    match &policy.expiration_action {
                        ExpirationAction::Delete => {
                            self.delete_pattern(&pattern).await?;
                            report.deleted += 1;
                        }
                        ExpirationAction::AnonymizeAndPromote => {
                            self.anonymize_and_promote(&pattern).await?;
                            report.promoted += 1;
                        }
                        ExpirationAction::Archive { storage_tier } => {
                            self.archive_pattern(&pattern, storage_tier).await?;
                            report.archived += 1;
                        }
                        ExpirationAction::ExtendIfActive { extension } => {
                            if pattern.recently_used() {
                                self.extend_retention(&pattern, *extension).await?;
                                report.extended += 1;
                            } else {
                                self.delete_pattern(&pattern).await?;
                                report.deleted += 1;
                            }
                        }
                    }
                }
            }
        }

        Ok(report)
    }
}
```

## Audit Logging

```rust
/// Privacy audit logging
pub struct AuditLog {
    /// Log storage
    storage: AuditLogStorage,
    /// Encryption for sensitive entries
    encryption: AuditEncryption,
}

impl AuditLog {
    /// Log consent event
    pub fn log_consent_granted(
        &mut self,
        tier: PrivacyTier,
        purpose: &ConsentPurpose,
        record: &ConsentRecord,
    ) {
        self.log(AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::ConsentGranted,
            tier,
            details: AuditDetails::Consent {
                purpose: purpose.clone(),
                scope: record.scope.clone(),
                expires: record.expires,
            },
            // Don't log user ID for privacy
            user_hash: self.hash_user_id(),
        });
    }

    /// Log data access
    pub fn log_data_access(
        &mut self,
        tier: PrivacyTier,
        pattern_id: PatternId,
        accessor: AccessorInfo,
        purpose: &str,
    ) {
        self.log(AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::DataAccess,
            tier,
            details: AuditDetails::Access {
                pattern_hash: hash_pattern_id(pattern_id),
                accessor_type: accessor.accessor_type,
                purpose: purpose.to_string(),
            },
            user_hash: self.hash_user_id(),
        });
    }

    /// Log federation participation
    pub fn log_federation_event(
        &mut self,
        round_id: RoundId,
        event: FederationEventType,
        patterns_contributed: usize,
    ) {
        self.log(AuditEntry {
            timestamp: Utc::now(),
            event_type: AuditEventType::Federation,
            tier: PrivacyTier::Public,
            details: AuditDetails::Federation {
                round_id,
                event,
                pattern_count: patterns_contributed,
            },
            user_hash: self.hash_user_id(),
        });
    }

    /// Generate privacy report for user
    pub async fn generate_privacy_report(&self) -> Result<PrivacyReport, AuditError> {
        let user_hash = self.hash_user_id();
        let entries = self.storage.query_by_user(&user_hash).await?;

        Ok(PrivacyReport {
            generated_at: Utc::now(),
            consent_history: self.extract_consent_history(&entries),
            data_access_summary: self.summarize_data_access(&entries),
            federation_participation: self.summarize_federation(&entries),
            data_retention_status: self.get_retention_status().await?,
        })
    }
}
```

## Configuration

```toml
# privacy.toml

[privacy]
# Default tier for new patterns
default_tier = "private"

[privacy.tiers.private]
encryption = "aes-256-gcm"
key_derivation = "device-bound"
sharing = "never"

[privacy.tiers.group]
encryption = "aes-256-gcm"
key_exchange = "x3dh"
max_group_size = 100

[privacy.tiers.tenant]
encryption = "aes-256-gcm"
key_management = "hsm"
access_control = "rbac"

[privacy.tiers.public]
anonymization = "full"
differential_privacy = true
epsilon = 1.0
delta = 1e-5
k_anonymity = 5

[privacy.consent]
require_explicit = true
allow_revocation = true
grace_period_days = 30

[privacy.retention]
private_max_days = 365
group_max_days = 180
tenant_max_days = 90
public_max_days = "indefinite"

[privacy.pii]
detect_emails = true
detect_phones = true
detect_names = true
ml_detection = true

[privacy.audit]
enabled = true
retention_days = 730
encrypt_sensitive = true
```

---

**Previous**: [04-FEDERATION.md](./04-FEDERATION.md) - Secure federated learning
**Next**: [06-SIMD-GPU.md](./06-SIMD-GPU.md) - Hardware acceleration
