import { describe, expect, it } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

import {
  addMetrics,
  dailyExportRows,
  emptyBreakdown,
  emptyMetrics,
  netRateOf,
  totalsFromResponse,
} from "./tokenAnalytics";

const metrics = (overrides: Partial<SpendMetrics> = {}): SpendMetrics => ({ ...emptyMetrics(), ...overrides });

const day = (date: string, overrides: Partial<SpendMetrics>): DailyData => ({
  date,
  metrics: metrics(overrides),
  breakdown: emptyBreakdown(),
});

describe("token analytics calculations", () => {
  it("preserves negative compression returns and calculates the signed net rate", () => {
    const result = metrics({
      prompt_tokens: 1_250,
      compression_gross_saved_tokens: 600,
      compression_extra_input_tokens: 850,
      compression_saved_tokens: -250,
    });

    expect(netRateOf(result)).toBe(-0.25);
    expect(result.compression_gross_saved_tokens! - result.compression_extra_input_tokens!).toBe(
      result.compression_saved_tokens,
    );
  });

  it("uses exact endpoint metadata totals without converting cache reads into token savings", () => {
    const totals = totalsFromResponse({
      results: [day("2026-08-13", { cache_read_input_tokens: 8_000, compression_saved_tokens: -250 })],
      metadata: {
        total_cache_read_input_tokens: 8_000,
        total_compression_saved_tokens: -250,
        total_compression_gross_saved_tokens: 600,
        total_compression_extra_input_tokens: 850,
        total_compression_requests: 1,
        total_compression_net_savings_rate: -0.25,
        total_prompt_caching_savings_spend: 0.012,
      },
    });

    expect(totals.total_cache_read_input_tokens).toBe(8_000);
    expect(totals.total_compression_saved_tokens).toBe(-250);
    expect(totals.total_prompt_caching_savings_spend).toBe(0.012);
  });

  it("exports gross, recall, net, cache, and request metrics as separate columns", () => {
    const rows = dailyExportRows({
      results: [
        day("2026-08-13", {
          api_requests: 4,
          compression_requests: 2,
          compression_gross_saved_tokens: 1_200,
          compression_extra_input_tokens: 950,
          compression_saved_tokens: 250,
          cache_read_input_tokens: 400,
        }),
      ],
      metadata: {},
    });

    expect(rows[0]).toMatchObject({
      请求数: 4,
      压缩次数: 2,
      毛节省Token: 1_200,
      召回额外输入Token: 950,
      净节省Token: 250,
      缓存读取Token: 400,
    });
    expect(rows[0]).not.toHaveProperty("缓存写入Token");
  });

  it("accumulates request and optimization metrics across rows", () => {
    const total = addMetrics(
      metrics({ api_requests: 1, compression_requests: 1, compression_saved_tokens: 500 }),
      metrics({ api_requests: 2, compression_requests: 1, compression_saved_tokens: -250 }),
    );

    expect(total.api_requests).toBe(3);
    expect(total.compression_requests).toBe(2);
    expect(total.compression_saved_tokens).toBe(250);
  });
});
