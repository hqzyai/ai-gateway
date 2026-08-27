import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders } from "../../../../../../tests/test-utils";
import { beforeEach, describe, expect, it, vi } from "vitest";
import PricingRulesManager from "./PricingRulesManager";
import { getModelCostMapOverrides } from "./api";

vi.mock("@/components/price_data_reload", () => ({
  default: () => <button>刷新最新价格</button>,
}));

vi.mock("./api", () => ({
  getModelCostMapOverrides: vi.fn(),
  upsertModelCostMapOverride: vi.fn(),
  deleteModelCostMapOverride: vi.fn(),
}));

describe("PricingRulesManager", () => {
  beforeEach(() => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({
      overrides: [
        {
          model_name: "volcengine/video-model",
          values: {
            litellm_provider: "volcengine",
            mode: "video_generation",
            video_token_pricing: {
              no_video_input: 0.000046,
              video_input_1080p: 0.000031,
            },
            tiered_pricing: [
              {
                range: [0, 32768],
                input_cost_per_token: 0.0000032,
                output_cost_per_token: 0.000016,
              },
            ],
            output_cost_per_second: 0.05,
            output_cost_per_second_720p: 0.05,
            output_cost_per_second_1080p: 0.075,
          },
        },
      ],
    });
  });

  it("opens a visual editor for video scenarios and token tiers", async () => {
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{
          "volcengine/video-model": {
            litellm_provider: "volcengine",
            mode: "video_generation",
            video_token_pricing: {
              no_video_input: 0.000046,
              video_input_1080p: 0.000031,
            },
            tiered_pricing: [
              {
                range: [0, 32768],
                input_cost_per_token: 0.0000032,
                output_cost_per_token: 0.000016,
              },
            ],
            output_cost_per_second: 0.05,
            output_cost_per_second_720p: 0.05,
            output_cost_per_second_1080p: 0.075,
          },
        }}
        loading={false}
        onModelCostMapReload={vi.fn()}
      />,
    );

    expect(await screen.findByText("volcengine/video-model")).toBeInTheDocument();
    expect(screen.getByText("自定义规则")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /配置/ }));
    expect(await screen.findByText("编辑自定义计费规则")).toBeInTheDocument();
    expect(screen.getAllByText("输出视频（默认分辨率）").length).toBeGreaterThan(0);
    expect(screen.getAllByText("输出视频 720P").length).toBeGreaterThan(0);
    expect(screen.getAllByText("输出视频 1080P").length).toBeGreaterThan(0);
    expect(screen.getByRole("tab", { name: "视频 Token（2）" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Token 阶梯（1）" })).toBeInTheDocument();
  });

  it("shows per-generation prices for 3D models without million-token scaling", async () => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({ overrides: [] });
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{
          "custom-3d-provider/model": {
            litellm_provider: "custom-3d-provider",
            mode: "video_generation",
            video_token_pricing_unit: "per_generation",
            video_token_pricing: {
              no_video_input_standard_no_texture: 0.7,
              video_input_standard_hd_texture: 2.8,
            },
          },
        }}
        loading={false}
        onModelCostMapReload={vi.fn()}
      />,
    );

    const row = (await screen.findByText("custom-3d-provider/model")).closest("tr");
    expect(row).not.toBeNull();
    const modelRow = within(row as HTMLElement);
    expect(modelRow.getByText("生成资产")).toBeInTheDocument();
    fireEvent.click(modelRow.getByRole("button", { name: /配置/ }));
    fireEvent.click(await screen.findByRole("tab", { name: "生成资产（2）" }));
    expect(screen.getByText("每个生成结果（3D / 资产）")).toBeInTheDocument();
    const prices = screen.getAllByRole("spinbutton").map((input) => Number((input as HTMLInputElement).value));
    expect(prices).toContain(0.7);
    expect(prices).toContain(2.8);
    expect(prices).not.toContain(700000);
  });
});
