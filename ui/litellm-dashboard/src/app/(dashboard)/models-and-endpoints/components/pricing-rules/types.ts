export type PricingValue = string | number | boolean | null | PricingValue[] | { [key: string]: PricingValue };

export type ModelCostEntry = Record<string, PricingValue>;

export interface ModelCostOverride {
  model_name: string;
  values: ModelCostEntry;
}

export interface ModelCostOverrideListResponse {
  overrides: ModelCostOverride[];
}

export interface ModelCostOverrideMutationResponse {
  status: string;
  model_name: string;
  override_count: number;
}

export interface PricingDimensionDefinition {
  key: string;
  label: string;
  category: string;
  unit: string;
  scale: number;
}

export interface PricingDimensionRow {
  id: string;
  key: string;
  value: number;
}

export interface VideoPricingRow {
  id: string;
  inputType: "video_input" | "no_video_input";
  resolution: string;
  value: number;
}

export interface ImagePricingRow {
  id: string;
  direction: "input" | "output";
  resolution: string;
  value: number;
}

export type VideoPricingUnit = "per_token" | "per_generation";

export interface TieredPricingRow {
  id: string;
  lower: number;
  upper: number | null;
  input: number | null;
  output: number | null;
  cacheRead: number | null;
}

export interface PricingRuleDraft {
  modelName: string;
  provider: string;
  mode: string;
  source: string;
  dimensions: PricingDimensionRow[];
  imagePricing: ImagePricingRow[];
  videoPricing: VideoPricingRow[];
  videoPricingUnit: VideoPricingUnit;
  tiers: TieredPricingRow[];
}
