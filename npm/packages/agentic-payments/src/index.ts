/**
 * @ruvector/agentic-payments
 *
 * High-performance payment processing for WASM App Store.
 * Supports credits, subscriptions, micropayments, and chip app monetization.
 */

// Types
export interface Account {
  id: string;
  external_id?: string;
  account_type: AccountType;
  status: AccountStatus;
  credits: CreditBalance;
  subscription?: Subscription;
  created_at: string;
  updated_at: string;
}

export type AccountType = 'user' | 'developer' | 'organization' | 'platform';
export type AccountStatus = 'active' | 'suspended' | 'pending' | 'closed';

export interface CreditBalance {
  account_id: string;
  available: number;
  reserved: number;
  lifetime_earned: number;
  lifetime_spent: number;
  updated_at: string;
}

export interface CreditTransaction {
  id: string;
  account_id: string;
  transaction_type: CreditTransactionType;
  amount: number;
  balance_after: number;
  description: string;
  app_id?: string;
  created_at: string;
}

export type CreditTransactionType =
  | 'purchase'
  | 'earned'
  | 'spent'
  | 'refund'
  | 'transfer'
  | 'bonus'
  | 'promo'
  | 'subscription'
  | 'challenge_reward'
  | 'referral_bonus'
  | 'revenue_share';

export interface Transaction {
  id: string;
  transaction_type: TransactionType;
  from_account: string;
  to_account?: string;
  amount: number;
  platform_fee: number;
  net_amount: number;
  app_id?: string;
  status: TransactionStatus;
  description: string;
  created_at: string;
  completed_at?: string;
}

export type TransactionType =
  | 'purchase'
  | 'app_payment'
  | 'subscription'
  | 'transfer'
  | 'refund'
  | 'payout'
  | 'bonus'
  | 'micropayment';

export type TransactionStatus =
  | 'pending'
  | 'processing'
  | 'completed'
  | 'failed'
  | 'cancelled'
  | 'refunded'
  | 'disputed';

export interface TransactionReceipt {
  transaction_id: string;
  receipt_number: string;
  amount: number;
  status: TransactionStatus;
  description: string;
  timestamp: string;
}

export interface Subscription {
  id: string;
  account_id: string;
  tier: SubscriptionTier;
  billing_period: BillingPeriod;
  status: SubscriptionStatus;
  monthly_credits: number;
  credits_remaining: number;
  current_period_start: string;
  current_period_end: string;
  created_at: string;
}

export type SubscriptionTier = 'free' | 'pro' | 'enterprise' | 'custom';
export type BillingPeriod = 'monthly' | 'annual';
export type SubscriptionStatus = 'active' | 'past_due' | 'cancelled' | 'expired' | 'trial' | 'paused';

export interface SubscriptionPlan {
  id: string;
  name: string;
  description: string;
  tier: SubscriptionTier;
  monthly_price_cents: number;
  annual_price_cents: number;
  monthly_credits: number;
  features: string[];
  is_popular: boolean;
}

export interface MicropaymentChannel {
  id: string;
  user_account: string;
  developer_account: string;
  app_id: string;
  deposited: number;
  spent: number;
  balance: number;
  payment_count: number;
  status: ChannelStatus;
  expires_at: string;
  created_at: string;
}

export type ChannelStatus = 'active' | 'closed' | 'expired' | 'settling';

export interface MicropaymentReceipt {
  id: string;
  channel_id: string;
  amount: number;
  total_spent: number;
  remaining_balance: number;
  payment_number: number;
  timestamp: string;
}

export interface CreditPricing {
  credits_per_dollar: number;
  min_purchase_cents: number;
  max_purchase_cents: number;
  bulk_discounts: BulkDiscount[];
}

export interface BulkDiscount {
  min_amount: number;
  bonus_percentage: number;
}

export interface PaymentEngineStats {
  total_accounts: number;
  total_developers: number;
  total_transactions: number;
  active_channels: number;
}

// Configuration
export interface PaymentEngineConfig {
  credit_pricing?: CreditPricing;
  platform_fee_percentage?: number;
  min_payout_amount?: number;
  channel_default_duration_hours?: number;
  enable_signatures?: boolean;
}

// Default configuration
export const DEFAULT_CONFIG: PaymentEngineConfig = {
  platform_fee_percentage: 25,
  min_payout_amount: 1000,
  channel_default_duration_hours: 24,
  enable_signatures: false,
};

// Default credit pricing
export const DEFAULT_CREDIT_PRICING: CreditPricing = {
  credits_per_dollar: 100,
  min_purchase_cents: 1000,
  max_purchase_cents: 1000000,
  bulk_discounts: [
    { min_amount: 5000, bonus_percentage: 5 },
    { min_amount: 10000, bonus_percentage: 10 },
    { min_amount: 50000, bonus_percentage: 15 },
    { min_amount: 100000, bonus_percentage: 20 },
  ],
};

// Default subscription plans
export const SUBSCRIPTION_PLANS: SubscriptionPlan[] = [
  {
    id: 'free',
    name: 'Free',
    description: 'Perfect for trying out chip apps',
    tier: 'free',
    monthly_price_cents: 0,
    annual_price_cents: 0,
    monthly_credits: 100,
    features: [
      '100 credits/month',
      '10 free chip app uses/day',
      'Community support',
      'Basic analytics',
    ],
    is_popular: false,
  },
  {
    id: 'pro',
    name: 'Pro',
    description: 'For developers and power users',
    tier: 'pro',
    monthly_price_cents: 2900,
    annual_price_cents: 29000,
    monthly_credits: 1000,
    features: [
      '1,000 credits/month',
      'Unlimited chip app uses',
      'Priority support',
      'Advanced analytics',
      '5 concurrent executions',
      'Revenue sharing for published apps',
    ],
    is_popular: true,
  },
  {
    id: 'enterprise',
    name: 'Enterprise',
    description: 'For teams and organizations',
    tier: 'enterprise',
    monthly_price_cents: 9900,
    annual_price_cents: 99000,
    monthly_credits: 10000,
    features: [
      '10,000 credits/month',
      'Unlimited everything',
      'Dedicated support',
      'Custom analytics',
      '50 concurrent executions',
      'Higher revenue share',
      'SLA guarantee',
      'Custom integrations',
    ],
    is_popular: false,
  },
];

// Utility functions
export function calculateCredits(amountCents: number, pricing: CreditPricing = DEFAULT_CREDIT_PRICING): number {
  const baseCredits = (amountCents * pricing.credits_per_dollar) / 100;
  const bonusPercentage = getBonusPercentage(amountCents, pricing);
  const bonusCredits = (baseCredits * bonusPercentage) / 100;
  return Math.floor(baseCredits + bonusCredits);
}

function getBonusPercentage(amountCents: number, pricing: CreditPricing): number {
  for (const discount of [...pricing.bulk_discounts].reverse()) {
    if (amountCents >= discount.min_amount) {
      return discount.bonus_percentage;
    }
  }
  return 0;
}

export function formatCredits(credits: number): string {
  if (credits >= 1000000) {
    return `${(credits / 1000000).toFixed(1)}M`;
  }
  if (credits >= 1000) {
    return `${(credits / 1000).toFixed(1)}K`;
  }
  return credits.toString();
}

export function creditsToDollars(credits: number): number {
  return credits / 100;
}

export function dollarsToCreditd(dollars: number): number {
  return dollars * 100;
}

// App size categories and pricing
export const APP_SIZE_PRICING = {
  chip: { maxBytes: 8 * 1024, basePrice: 1 },      // 8KB, 1 cent
  micro: { maxBytes: 64 * 1024, basePrice: 5 },    // 64KB, 5 cents
  small: { maxBytes: 512 * 1024, basePrice: 10 },  // 512KB, 10 cents
  medium: { maxBytes: 2 * 1024 * 1024, basePrice: 25 }, // 2MB, 25 cents
  large: { maxBytes: 10 * 1024 * 1024, basePrice: 50 }, // 10MB, 50 cents
  full: { maxBytes: Infinity, basePrice: 100 },    // >10MB, $1
} as const;

export function getAppSizeCategory(bytes: number): keyof typeof APP_SIZE_PRICING {
  if (bytes <= APP_SIZE_PRICING.chip.maxBytes) return 'chip';
  if (bytes <= APP_SIZE_PRICING.micro.maxBytes) return 'micro';
  if (bytes <= APP_SIZE_PRICING.small.maxBytes) return 'small';
  if (bytes <= APP_SIZE_PRICING.medium.maxBytes) return 'medium';
  if (bytes <= APP_SIZE_PRICING.large.maxBytes) return 'large';
  return 'full';
}

// Re-export WASM module loading helper
export async function loadWasmPaymentEngine(): Promise<typeof import('../wasm/agentic_payments_wasm')> {
  const wasm = await import('../wasm/agentic_payments_wasm');
  await wasm.default();
  return wasm;
}
