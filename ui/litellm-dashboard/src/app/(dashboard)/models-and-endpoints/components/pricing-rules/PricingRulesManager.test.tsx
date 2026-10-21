import { fireEvent, screen, within } from "@testing-library/react";
import { renderWithProviders, testQueryClient } from "../../../../../../tests/test-utils";
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
    testQueryClient.clear();
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

  it("scopes rows, counts and provider filters even when the backend returns every provider", async () => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({
      overrides: [
        {
          model_name: "openai/manual-model",
          values: { litellm_provider: "openai", mode: "chat", input_cost_per_token: 2e-6 },
        },
        {
          model_name: "anthropic/custom-model",
          values: { litellm_provider: "anthropic", mode: "chat", input_cost_per_token: 1e-6 },
        },
      ],
    });
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{
          "siliconflow/selected-model": { litellm_provider: "siliconflow", mode: "chat", input_cost_per_token: 0 },
          "volcengine/selected-model": { litellm_provider: "volcengine", mode: "chat", input_cost_per_token: 1e-6 },
          "dashscope/selected-model": {
            litellm_provider: "dashscope",
            mode: "image_generation",
            output_cost_per_image: 0.1,
          },
          "openai/builtin-model": { litellm_provider: "openai", mode: "chat", input_cost_per_token: 1e-6 },
          "bedrock/hidden-model": { litellm_provider: "bedrock", mode: "embedding", input_cost_per_token: 1e-6 },
          sample_spec: { litellm_provider: "dashscope", input_cost_per_token: 0 },
          "internal-id": { id: "internal-id", litellm_provider: "siliconflow", input_cost_per_token: 1e-6 },
          "openai/*": {},
        }}
        loading={false}
        onModelCostMapReload={vi.fn()}
      />,
    );

    expect(await screen.findByText("openai/manual-model")).toBeInTheDocument();
    for (const provider of ["siliconflow", "volcengine", "dashscope"]) {
      expect(screen.getByText(`${provider}/selected-model`)).toBeInTheDocument();
    }
    for (const name of [
      "openai/builtin-model",
      "bedrock/hidden-model",
      "anthropic/custom-model",
      "sample_spec",
      "internal-id",
      "openai/*",
    ]) {
      expect(screen.queryByText(name)).not.toBeInTheDocument();
    }
    expect(screen.getByText("共 4 个模型")).toBeInTheDocument();
    const pricedCard = screen.getByText("已有价格的模型").closest(".ant-card") as HTMLElement;
    const customCard = screen.getByText("自定义规则").closest(".ant-card") as HTMLElement;
    const modesCard = screen.getByText("计费模式").closest(".ant-card") as HTMLElement;
    expect(within(pricedCard).getByText("4")).toBeInTheDocument();
    expect(within(customCard).getByText("1")).toBeInTheDocument();
    expect(within(modesCard).getByText("2")).toBeInTheDocument();
    fireEvent.mouseDown(screen.getAllByRole("combobox")[0]);
    const dropdown = document.querySelector(".ant-select-dropdown") as HTMLElement;
    for (const provider of ["siliconflow", "volcengine", "dashscope", "openai"]) {
      expect(within(dropdown).getByText(provider, { selector: ".ant-select-item-option-content" })).toBeInTheDocument();
    }
    expect(within(dropdown).queryByText("bedrock")).not.toBeInTheDocument();
    expect(within(dropdown).queryByText("anthropic")).not.toBeInTheDocument();
  });

  it("does not copy OpenAI upstream prices into a manually added rule", async () => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({
      overrides: [{ model_name: "openai/manual-model", values: { input_cost_per_token: 2e-6 } }],
    });
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{
          "openai/manual-model": {
            litellm_provider: "openai",
            mode: "chat",
            input_cost_per_token: 1e-6,
            output_cost_per_token: 9e-6,
          },
        }}
        loading={false}
        onModelCostMapReload={vi.fn()}
      />,
    );

    const row = (await screen.findByText("openai/manual-model")).closest("tr") as HTMLElement;
    fireEvent.click(within(row).getByRole("button", { name: /配置/ }));
    expect(await screen.findByText("编辑自定义计费规则")).toBeInTheDocument();
    const prices = screen.getAllByRole("spinbutton").map((input) => Number((input as HTMLInputElement).value));
    expect(prices).toEqual([2]);
  });

  it("only offers the four selected providers when adding a pricing rule", async () => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({ overrides: [] });
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{}}
        loading={false}
        onModelCostMapReload={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /新增计费规则/ }));
    const drawer = await screen.findByRole("dialog");
    expect(within(drawer).getByText("siliconflow")).toBeInTheDocument();
    expect(within(drawer).queryByText("custom")).not.toBeInTheDocument();
    fireEvent.mouseDown(within(drawer).getAllByRole("combobox")[0]);
    const dropdown = document.querySelector(".ant-select-dropdown") as HTMLElement;
    const options = Array.from(
      dropdown.querySelectorAll(".ant-select-item-option-content"),
      (item) => item.textContent,
    );
    expect(options).toEqual(["siliconflow", "volcengine", "dashscope", "openai"]);
  });

  it("shows per-generation prices for 3D models without million-token scaling", async () => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({ overrides: [] });
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{
          "siliconflow/asset-test-model": {
            litellm_provider: "siliconflow",
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

    const row = (await screen.findByText("siliconflow/asset-test-model")).closest("tr");
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

  it("shows editable image resolution prices", async () => {
    vi.mocked(getModelCostMapOverrides).mockResolvedValue({ overrides: [] });
    renderWithProviders(
      <PricingRulesManager
        accessToken="test-token"
        userRole="Admin"
        modelCostMap={{
          "dashscope/qwen-image-3.0-pro": {
            litellm_provider: "dashscope",
            mode: "image_generation",
            image_resolution_pricing: {
              input_1k: 0.02,
              output_1k: 0.25,
              output_2k: 0.5,
            },
          },
        }}
        loading={false}
        onModelCostMapReload={vi.fn()}
      />,
    );

    const row = (await screen.findByText("dashscope/qwen-image-3.0-pro")).closest("tr");
    expect(row).not.toBeNull();
    const modelRow = within(row as HTMLElement);
    expect(modelRow.getByText("图片分辨率")).toBeInTheDocument();
    fireEvent.click(modelRow.getByRole("button", { name: /配置/ }));
    fireEvent.click(await screen.findByRole("tab", { name: "图片分辨率（3）" }));
    expect(
      screen.getByText("按输入 / 输出方向和分辨率档配置每张图片的价格。分辨率档可填写 1k、2k、4k 等供应商返回的标识。"),
    ).toBeInTheDocument();
    const prices = screen.getAllByRole("spinbutton").map((input) => Number((input as HTMLInputElement).value));
    expect(prices).toContain(0.02);
    expect(prices).toContain(0.25);
    expect(prices).toContain(0.5);
  });
});
