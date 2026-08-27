import {
  ModelCostEntry,
  PricingDimensionDefinition,
  PricingDimensionRow,
  PricingRuleDraft,
  PricingValue,
  TieredPricingRow,
  VideoPricingRow,
  VideoPricingUnit,
} from "./types";

const perMillion = 1_000_000;
const storedPrice = (value: number, scale: number): number => Number((value / scale).toPrecision(15));

export const getVideoPricingUnit = (entry: ModelCostEntry): VideoPricingUnit =>
  entry.video_token_pricing_unit === "per_generation" ? "per_generation" : "per_token";

export const getVideoPricingScale = (unit: VideoPricingUnit): number => (unit === "per_generation" ? 1 : perMillion);

const modeLabels: Readonly<Record<string, string>> = Object.freeze({
  chat: "对话",
  responses: "响应",
  embedding: "向量嵌入",
  image_generation: "图像生成",
  video_generation: "视频生成",
  audio_speech: "语音合成",
  audio_transcription: "语音转写",
  rerank: "重排序",
  search: "搜索",
  ocr: "OCR",
  unknown: "未知",
});

export const getModeLabel = (mode: string): string => modeLabels[mode] ?? mode.replaceAll("_", " ");

export const pricingDimensions: PricingDimensionDefinition[] = [
  {
    key: "input_cost_per_token",
    label: "输入文本 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_token",
    label: "输出文本 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "cache_read_input_token_cost",
    label: "缓存读取 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "cache_creation_input_token_cost",
    label: "缓存写入 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_reasoning_token",
    label: "推理 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "input_cost_per_audio_token",
    label: "输入音频 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_audio_token",
    label: "输出音频 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "input_cost_per_image_token",
    label: "输入图像 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_image_token",
    label: "输出图像 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "input_cost_per_video_token",
    label: "输入视频 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_video_token",
    label: "输出视频 Token",
    category: "Token 计费",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "input_cost_per_token_batches",
    label: "批量输入 Token",
    category: "批量与优先级",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_token_batches",
    label: "批量输出 Token",
    category: "批量与优先级",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "input_cost_per_token_priority",
    label: "优先级输入 Token",
    category: "批量与优先级",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_token_priority",
    label: "优先级输出 Token",
    category: "批量与优先级",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "input_cost_per_token_flex",
    label: "弹性输入 Token",
    category: "批量与优先级",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  {
    key: "output_cost_per_token_flex",
    label: "弹性输出 Token",
    category: "批量与优先级",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
  { key: "input_cost_per_image", label: "输入图像", category: "图像", unit: "美元 / 张", scale: 1 },
  { key: "output_cost_per_image", label: "输出图像", category: "图像", unit: "美元 / 张", scale: 1 },
  {
    key: "output_cost_per_image_above_16384_tokens",
    label: "输出图像（超过 16384 Token）",
    category: "图像",
    unit: "美元 / 张",
    scale: 1,
  },
  { key: "input_cost_per_pixel", label: "输入像素", category: "图像", unit: "美元 / 像素", scale: 1 },
  { key: "output_cost_per_pixel", label: "输出像素", category: "图像", unit: "美元 / 像素", scale: 1 },
  {
    key: "input_cost_per_video_per_second",
    label: "输入视频（按时长）",
    category: "视频按秒",
    unit: "美元 / 秒",
    scale: 1,
  },
  {
    key: "output_cost_per_video_per_second",
    label: "输出视频（按时长）",
    category: "视频按秒",
    unit: "美元 / 秒",
    scale: 1,
  },
  {
    key: "output_cost_per_second",
    label: "输出视频（默认分辨率）",
    category: "视频按秒",
    unit: "美元 / 秒",
    scale: 1,
  },
  {
    key: "output_cost_per_second_480p",
    label: "输出视频 480P",
    category: "视频按秒",
    unit: "美元 / 秒",
    scale: 1,
  },
  {
    key: "output_cost_per_second_720p",
    label: "输出视频 720P",
    category: "视频按秒",
    unit: "美元 / 秒",
    scale: 1,
  },
  {
    key: "output_cost_per_second_1080p",
    label: "输出视频 1080P",
    category: "视频按秒",
    unit: "美元 / 秒",
    scale: 1,
  },
  { key: "input_cost_per_audio_per_second", label: "输入音频", category: "音频", unit: "美元 / 秒", scale: 1 },
  { key: "output_cost_per_audio_per_second", label: "输出音频", category: "音频", unit: "美元 / 秒", scale: 1 },
  { key: "input_cost_per_second", label: "输入时长", category: "音频", unit: "美元 / 秒", scale: 1 },
  {
    key: "input_cost_per_character",
    label: "输入字符",
    category: "文本与请求",
    unit: "美元 / 字符",
    scale: 1,
  },
  {
    key: "output_cost_per_character",
    label: "输出字符",
    category: "文本与请求",
    unit: "美元 / 字符",
    scale: 1,
  },
  { key: "input_cost_per_query", label: "输入请求", category: "文本与请求", unit: "美元 / 次", scale: 1 },
  {
    key: "search_context_cost_per_query",
    label: "搜索上下文",
    category: "文本与请求",
    unit: "美元 / 次",
    scale: 1,
  },
  { key: "ocr_cost_per_page", label: "OCR 页", category: "文档", unit: "美元 / 页", scale: 1 },
  { key: "ocr_cost_per_credit", label: "OCR 额度", category: "文档", unit: "美元 / 额度", scale: 1 },
  {
    key: "citation_cost_per_token",
    label: "引用 Token",
    category: "文本与请求",
    unit: "美元 / 百万 Token",
    scale: perMillion,
  },
];

const definitionsByKey = new Map(pricingDimensions.map((definition) => [definition.key, definition]));

export const getPricingDefinition = (key: string): PricingDimensionDefinition =>
  definitionsByKey.get(key) ?? {
    key,
    label: key.replaceAll("_", " "),
    category: "自定义",
    unit: "美元 / 单位",
    scale: key.includes("token") ? perMillion : 1,
  };

const isPricingNumber = (key: string, value: PricingValue): value is number => {
  const hasPricingKey = ["cost", "price", "pricing"].some((fragment) => key.includes(fragment));
  return typeof value === "number" && hasPricingKey;
};

const createId = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

const parseVideoPricing = (value: PricingValue, unit: VideoPricingUnit): VideoPricingRow[] => {
  if (value === null || Array.isArray(value) || typeof value !== "object") return [];
  return Object.entries(value).flatMap(([key, price]) => {
    if (typeof price !== "number") return [];
    const inputType = key.startsWith("no_video_input") ? "no_video_input" : "video_input";
    const resolution = key.replace(inputType, "").replace(/^_/, "") || "default";
    return [{ id: createId(), inputType, resolution, value: price * getVideoPricingScale(unit) }];
  });
};

const optionalScaledNumber = (value: PricingValue | undefined): number | null =>
  typeof value === "number" ? value * perMillion : null;

const parseTiers = (value: PricingValue): TieredPricingRow[] => {
  if (!Array.isArray(value)) return [];
  return value.flatMap((tier) => {
    if (tier === null || Array.isArray(tier) || typeof tier !== "object") return [];
    const range = tier.range;
    if (!Array.isArray(range) || typeof range[0] !== "number") return [];
    return [
      {
        id: createId(),
        lower: range[0],
        upper: typeof range[1] === "number" ? range[1] : null,
        input: optionalScaledNumber(tier.input_cost_per_token),
        output: optionalScaledNumber(tier.output_cost_per_token),
        cacheRead: optionalScaledNumber(tier.cache_read_input_token_cost),
      },
    ];
  });
};

export const createPricingRuleDraft = (modelName: string, entry: ModelCostEntry): PricingRuleDraft => {
  const videoPricingUnit = getVideoPricingUnit(entry);
  return {
    modelName,
    provider: typeof entry.litellm_provider === "string" ? entry.litellm_provider : modelName.split("/")[0] || "custom",
    mode: typeof entry.mode === "string" ? entry.mode : "chat",
    source: typeof entry.source === "string" ? entry.source : "",
    dimensions: Object.entries(entry).flatMap(([key, value]): PricingDimensionRow[] => {
      if (key === "video_token_pricing" || key === "tiered_pricing" || !isPricingNumber(key, value)) return [];
      return [{ id: createId(), key, value: value * getPricingDefinition(key).scale }];
    }),
    videoPricing: parseVideoPricing(entry.video_token_pricing, videoPricingUnit),
    videoPricingUnit,
    tiers: parseTiers(entry.tiered_pricing),
  };
};

const nestedPricingKey = (row: VideoPricingRow) =>
  `${row.inputType}${row.resolution === "default" ? "" : `_${row.resolution.trim().toLowerCase()}`}`;

const sortedTiers = (tiers: readonly TieredPricingRow[]): readonly TieredPricingRow[] =>
  [...tiers].sort((left, right) => left.lower - right.lower);

const flatTierPricing = (tiers: readonly TieredPricingRow[]): ModelCostEntry =>
  Object.fromEntries(
    sortedTiers(tiers).flatMap((tier, index) => {
      const suffix = index === 0 ? "" : `_above_${tier.lower}_tokens`;
      return [
        ...(tier.input === null
          ? []
          : [[`input_cost_per_token${suffix}`, storedPrice(tier.input, perMillion)] as const]),
        ...(tier.output === null
          ? []
          : [[`output_cost_per_token${suffix}`, storedPrice(tier.output, perMillion)] as const]),
        ...(tier.cacheRead === null
          ? []
          : [[`cache_read_input_token_cost${suffix}`, storedPrice(tier.cacheRead, perMillion)] as const]),
      ];
    }),
  );

export const serializePricingRule = (draft: PricingRuleDraft): ModelCostEntry => ({
  litellm_provider: draft.provider,
  mode: draft.mode,
  ...(draft.source ? { source: draft.source } : {}),
  ...Object.fromEntries(
    draft.dimensions.map((row) => [row.key, storedPrice(row.value, getPricingDefinition(row.key).scale)]),
  ),
  ...flatTierPricing(draft.tiers),
  ...(draft.videoPricing.length > 0
    ? {
        video_token_pricing: Object.fromEntries(
          draft.videoPricing.map((row) => [
            nestedPricingKey(row),
            storedPrice(row.value, getVideoPricingScale(draft.videoPricingUnit)),
          ]),
        ),
        ...(draft.videoPricingUnit === "per_generation" ? { video_token_pricing_unit: draft.videoPricingUnit } : {}),
      }
    : {}),
  ...(draft.tiers.length > 0
    ? {
        tiered_pricing: sortedTiers(draft.tiers).map((tier) => ({
          range: [tier.lower, tier.upper ?? Number.MAX_SAFE_INTEGER],
          ...(tier.input === null ? {} : { input_cost_per_token: storedPrice(tier.input, perMillion) }),
          ...(tier.output === null ? {} : { output_cost_per_token: storedPrice(tier.output, perMillion) }),
          ...(tier.cacheRead === null ? {} : { cache_read_input_token_cost: storedPrice(tier.cacheRead, perMillion) }),
        })),
      }
    : {}),
});

export const pricingKeysForEntry = (entry: ModelCostEntry): string[] =>
  Object.entries(entry)
    .filter(([key, value]) => key === "video_token_pricing" || key === "tiered_pricing" || isPricingNumber(key, value))
    .map(([key]) => key);
