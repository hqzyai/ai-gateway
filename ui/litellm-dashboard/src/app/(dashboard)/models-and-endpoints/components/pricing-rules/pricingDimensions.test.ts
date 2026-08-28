import { describe, expect, it } from "vitest";
import { createPricingRuleDraft, serializePricingRule } from "./pricingDimensions";
import { ModelCostEntry } from "./types";

describe("pricing dimension serialization", () => {
  it("round trips flat, tiered, and resolution-aware video pricing", () => {
    const entry: ModelCostEntry = {
      litellm_provider: "volcengine",
      mode: "video_generation",
      source: "https://example.com/pricing",
      input_cost_per_token: 0.0000032,
      output_cost_per_token: 0.000016,
      cache_read_input_token_cost: 6.4e-7,
      output_cost_per_second: 0.05,
      output_cost_per_second_720p: 0.05,
      output_cost_per_second_1080p: 0.075,
      input_cost_per_token_above_32768_tokens: 0.0000048,
      output_cost_per_token_above_32768_tokens: 0.000024,
      cache_read_input_token_cost_above_32768_tokens: 9.6e-7,
      video_token_pricing: {
        no_video_input: 0.000046,
        no_video_input_1080p: 0.000051,
        video_input_4k: 0.000016,
      },
      tiered_pricing: [
        {
          range: [0, 32768],
          input_cost_per_token: 0.0000032,
          output_cost_per_token: 0.000016,
          cache_read_input_token_cost: 6.4e-7,
        },
        {
          range: [32768, 131072],
          input_cost_per_token: 0.0000048,
          output_cost_per_token: 0.000024,
        },
      ],
    };

    const serialized = serializePricingRule(createPricingRuleDraft("volcengine/model", entry));

    expect(serialized).toEqual(entry);
  });

  it("round trips per-generation 3D pricing without token scaling", () => {
    const entry: ModelCostEntry = {
      litellm_provider: "custom-3d-provider",
      mode: "video_generation",
      source: "https://example.com/pricing",
      video_token_pricing_unit: "per_generation",
      video_token_pricing: {
        no_video_input_standard_no_texture: 0.7,
        video_input_standard_hd_texture: 2.8,
      },
    };

    const draft = createPricingRuleDraft("custom-3d-provider/model", entry);

    expect(draft.videoPricingUnit).toBe("per_generation");
    expect(draft.videoPricing.map(({ inputType, resolution, value }) => ({ inputType, resolution, value }))).toEqual([
      { inputType: "no_video_input", resolution: "standard_no_texture", value: 0.7 },
      { inputType: "video_input", resolution: "standard_hd_texture", value: 2.8 },
    ]);
    expect(serializePricingRule(draft)).toEqual(entry);
  });

  it("round trips per-image resolution pricing without scaling", () => {
    const entry: ModelCostEntry = {
      litellm_provider: "dashscope",
      mode: "image_generation",
      image_resolution_pricing: {
        input_1k: 0.02,
        input_2k: 0.02,
        output_1k: 0.25,
        output_2k: 0.5,
      },
    };

    const draft = createPricingRuleDraft("dashscope/qwen-image-3.0-pro", entry);

    expect(draft.imagePricing.map(({ direction, resolution, value }) => ({ direction, resolution, value }))).toEqual([
      { direction: "input", resolution: "1k", value: 0.02 },
      { direction: "input", resolution: "2k", value: 0.02 },
      { direction: "output", resolution: "1k", value: 0.25 },
      { direction: "output", resolution: "2k", value: 0.5 },
    ]);
    expect(serializePricingRule(draft)).toEqual(entry);
  });

  it("serializes an open-ended final tier with a numeric upper bound", () => {
    const initialDraft = createPricingRuleDraft("custom/model", {
      tiered_pricing: [{ range: [0, 1000], input_cost_per_token: 0.000001 }],
    });
    const draft = {
      ...initialDraft,
      tiers: initialDraft.tiers.map((tier) => ({ ...tier, upper: null })),
    };

    const serialized = serializePricingRule(draft);
    const tiers = serialized.tiered_pricing;

    expect(Array.isArray(tiers)).toBe(true);
    if (!Array.isArray(tiers)) throw new Error("Expected tiered pricing array");
    const tier = tiers[0];
    if (tier === null || Array.isArray(tier) || typeof tier !== "object") throw new Error("Expected pricing tier");
    expect(tier.range).toEqual([0, Number.MAX_SAFE_INTEGER]);
  });

  it("generates the flat threshold fields used by the runtime cost calculator", () => {
    const initialDraft = createPricingRuleDraft("volcengine/custom-chat-model", {
      litellm_provider: "volcengine",
      mode: "chat",
    });
    const draft = {
      ...initialDraft,
      tiers: [
        {
          id: "high",
          lower: 32768,
          upper: null,
          input: 4.8,
          output: 24,
          cacheRead: 0.96,
        },
        {
          id: "base",
          lower: 0,
          upper: 32768,
          input: 3.2,
          output: 16,
          cacheRead: 0.64,
        },
      ],
    };

    const serialized = serializePricingRule(draft);

    expect(serialized).toMatchObject({
      input_cost_per_token: 0.0000032,
      output_cost_per_token: 0.000016,
      cache_read_input_token_cost: 6.4e-7,
      input_cost_per_token_above_32768_tokens: 0.0000048,
      output_cost_per_token_above_32768_tokens: 0.000024,
      cache_read_input_token_cost_above_32768_tokens: 9.6e-7,
      tiered_pricing: [
        {
          range: [0, 32768],
          input_cost_per_token: 0.0000032,
          output_cost_per_token: 0.000016,
          cache_read_input_token_cost: 6.4e-7,
        },
        {
          range: [32768, Number.MAX_SAFE_INTEGER],
          input_cost_per_token: 0.0000048,
          output_cost_per_token: 0.000024,
          cache_read_input_token_cost: 9.6e-7,
        },
      ],
    });
  });
});
