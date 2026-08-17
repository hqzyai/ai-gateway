import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";

import { emptyBreakdown, emptyMetrics, type TokenAnalyticsResponse } from "./tokenAnalytics";

const analyticsCall = vi.fn();

vi.mock("@/components/networking", () => ({
  userDailyActivityAggregatedFilteredCall: (...args: unknown[]) => analyticsCall(...args),
}));

vi.mock("@/components/shared/charts", () => ({
  AreaChart: ({ categories }: { categories: string[] }) => <div data-testid="chart">{categories.join(",")}</div>,
}));

const metrics = (overrides: Partial<SpendMetrics> = {}): SpendMetrics => ({ ...emptyMetrics(), ...overrides });

const response = (): TokenAnalyticsResponse => {
  const modelMetrics = metrics({ api_requests: 2, total_tokens: 1_260, compression_saved_tokens: -250 });
  const keyMetrics = metrics({ api_requests: 2, total_tokens: 1_260, compression_saved_tokens: -250 });
  const breakdown = emptyBreakdown();
  breakdown.models["gpt-4o"] = { metrics: modelMetrics, metadata: {}, api_key_breakdown: {} };
  breakdown.api_keys["key-hash-1234567890"] = {
    metrics: keyMetrics,
    metadata: { key_alias: "production-key", team_id: null },
  };
  const daily: DailyData = {
    date: "2026-08-13",
    metrics: metrics({
      prompt_tokens: 1_250,
      completion_tokens: 10,
      total_tokens: 1_260,
      api_requests: 2,
      successful_requests: 2,
      compression_requests: 1,
      compression_gross_saved_tokens: 600,
      compression_extra_input_tokens: 850,
      compression_saved_tokens: -250,
      cache_read_input_tokens: 321,
    }),
    breakdown,
  };
  return {
    results: [daily],
    metadata: {
      total_prompt_tokens: 1_250,
      total_completion_tokens: 10,
      total_tokens: 1_260,
      total_api_requests: 2,
      total_successful_requests: 2,
      total_failed_requests: 0,
      total_compression_requests: 1,
      total_compression_gross_saved_tokens: 600,
      total_compression_extra_input_tokens: 850,
      total_compression_saved_tokens: -250,
      total_compression_net_savings_rate: -0.25,
      total_cache_read_input_tokens: 321,
    },
  };
};

import TokenAnalyticsView from "./TokenAnalyticsView";

describe("TokenAnalyticsView", () => {
  beforeEach(() => {
    analyticsCall.mockReset();
    analyticsCall.mockResolvedValue(response());
  });

  it("shows signed compression economics and keeps cache tokens separate", async () => {
    render(<TokenAnalyticsView accessToken="token" userId={null} />);

    await waitFor(() => expect(screen.getByText("-25.00%")).toBeInTheDocument());
    expect(screen.getByText("毛节省 Token")).toBeInTheDocument();
    expect(screen.getByText("召回额外输入 Token")).toBeInTheDocument();
    expect(screen.getByText("净节省 Token")).toBeInTheDocument();
    expect(screen.getAllByText("缓存读取 Token")).not.toHaveLength(0);
    expect(screen.queryByText("缓存写入 Token")).not.toBeInTheDocument();
    expect(screen.getByText("按厂商缓存折扣价计费，并非完全免费的 Token")).toBeInTheDocument();
  });

  it("sends API key and model together when both filters are selected", async () => {
    render(<TokenAnalyticsView accessToken="token" userId={null} />);
    await waitFor(() => expect(analyticsCall).toHaveBeenCalledTimes(2));
    analyticsCall.mockClear();

    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "manual-key-hash" } });
    fireEvent.change(screen.getByLabelText("模型"), { target: { value: "manual-model" } });
    expect(analyticsCall).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "应用筛选" }));

    await waitFor(() =>
      expect(analyticsCall).toHaveBeenCalledWith("token", expect.any(Date), expect.any(Date), {
        userId: null,
        apiKey: "manual-key-hash",
        model: "manual-model",
      }),
    );
  });
});
