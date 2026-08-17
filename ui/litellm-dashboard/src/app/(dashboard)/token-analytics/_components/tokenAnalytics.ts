import type { BreakdownMetrics, DailyData, KeyMetadata, SpendMetrics } from "@/components/UsagePage/types";

export interface TokenAnalyticsMetadata {
  total_spend: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  total_api_requests: number;
  total_successful_requests: number;
  total_failed_requests: number;
  total_cache_read_input_tokens: number;
  total_cache_creation_input_tokens: number;
  total_compression_saved_tokens: number;
  total_compression_gross_saved_tokens: number;
  total_compression_extra_input_tokens: number;
  total_compression_requests: number;
  total_compression_net_savings_rate: number;
  total_compression_savings_spend: number;
  total_compression_gross_savings_spend: number;
  total_compression_extra_input_spend: number;
  total_prompt_caching_savings_spend: number;
}

export interface TokenAnalyticsResponse {
  results: DailyData[];
  metadata: Partial<TokenAnalyticsMetadata>;
}

export interface DimensionRow extends SpendMetrics {
  id: string;
  label: string;
  keyMetadata?: KeyMetadata;
}

const numberOf = (value: number | null | undefined): number => value ?? 0;
const metadataOr = (value: number | undefined, fallback: number): number => (value === undefined ? fallback : value);

export const emptyBreakdown = (): BreakdownMetrics => ({
  models: {},
  model_groups: {},
  mcp_servers: {},
  providers: {},
  api_keys: {},
  entities: {},
  endpoints: {},
});

export const emptyMetrics = (): SpendMetrics => ({
  spend: 0,
  prompt_tokens: 0,
  completion_tokens: 0,
  total_tokens: 0,
  api_requests: 0,
  successful_requests: 0,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
  compression_saved_tokens: 0,
  compression_gross_saved_tokens: 0,
  compression_extra_input_tokens: 0,
  compression_requests: 0,
  compression_savings_spend: 0,
  compression_gross_savings_spend: 0,
  compression_extra_input_spend: 0,
  prompt_caching_savings_spend: 0,
});

export const addMetrics = (left: SpendMetrics, right: SpendMetrics): SpendMetrics => ({
  spend: left.spend + right.spend,
  prompt_tokens: left.prompt_tokens + right.prompt_tokens,
  completion_tokens: left.completion_tokens + right.completion_tokens,
  total_tokens: left.total_tokens + right.total_tokens,
  api_requests: left.api_requests + right.api_requests,
  successful_requests: left.successful_requests + right.successful_requests,
  failed_requests: left.failed_requests + right.failed_requests,
  cache_read_input_tokens: left.cache_read_input_tokens + right.cache_read_input_tokens,
  cache_creation_input_tokens: left.cache_creation_input_tokens + right.cache_creation_input_tokens,
  compression_saved_tokens: numberOf(left.compression_saved_tokens) + numberOf(right.compression_saved_tokens),
  compression_gross_saved_tokens:
    numberOf(left.compression_gross_saved_tokens) + numberOf(right.compression_gross_saved_tokens),
  compression_extra_input_tokens:
    numberOf(left.compression_extra_input_tokens) + numberOf(right.compression_extra_input_tokens),
  compression_requests: numberOf(left.compression_requests) + numberOf(right.compression_requests),
  compression_savings_spend: numberOf(left.compression_savings_spend) + numberOf(right.compression_savings_spend),
  compression_gross_savings_spend:
    numberOf(left.compression_gross_savings_spend) + numberOf(right.compression_gross_savings_spend),
  compression_extra_input_spend:
    numberOf(left.compression_extra_input_spend) + numberOf(right.compression_extra_input_spend),
  prompt_caching_savings_spend:
    numberOf(left.prompt_caching_savings_spend) + numberOf(right.prompt_caching_savings_spend),
});

export const totalsFromResponse = (response: TokenAnalyticsResponse | null): TokenAnalyticsMetadata => {
  const resultTotals = (response?.results ?? []).reduce((total, day) => addMetrics(total, day.metrics), emptyMetrics());
  const metadata = response?.metadata ?? {};
  const net = metadata.total_compression_saved_tokens ?? numberOf(resultTotals.compression_saved_tokens);
  const baseline = resultTotals.prompt_tokens + net;
  return {
    total_spend: metadataOr(metadata.total_spend, resultTotals.spend),
    total_prompt_tokens: metadataOr(metadata.total_prompt_tokens, resultTotals.prompt_tokens),
    total_completion_tokens: metadataOr(metadata.total_completion_tokens, resultTotals.completion_tokens),
    total_tokens: metadataOr(metadata.total_tokens, resultTotals.total_tokens),
    total_api_requests: metadataOr(metadata.total_api_requests, resultTotals.api_requests),
    total_successful_requests: metadataOr(metadata.total_successful_requests, resultTotals.successful_requests),
    total_failed_requests: metadataOr(metadata.total_failed_requests, resultTotals.failed_requests),
    total_cache_read_input_tokens: metadataOr(
      metadata.total_cache_read_input_tokens,
      resultTotals.cache_read_input_tokens,
    ),
    total_cache_creation_input_tokens: metadataOr(
      metadata.total_cache_creation_input_tokens,
      resultTotals.cache_creation_input_tokens,
    ),
    total_compression_saved_tokens: net,
    total_compression_gross_saved_tokens: metadataOr(
      metadata.total_compression_gross_saved_tokens,
      numberOf(resultTotals.compression_gross_saved_tokens),
    ),
    total_compression_extra_input_tokens: metadataOr(
      metadata.total_compression_extra_input_tokens,
      numberOf(resultTotals.compression_extra_input_tokens),
    ),
    total_compression_requests: metadataOr(
      metadata.total_compression_requests,
      numberOf(resultTotals.compression_requests),
    ),
    total_compression_net_savings_rate: metadataOr(
      metadata.total_compression_net_savings_rate,
      baseline > 0 ? net / baseline : 0,
    ),
    total_compression_savings_spend: metadataOr(
      metadata.total_compression_savings_spend,
      numberOf(resultTotals.compression_savings_spend),
    ),
    total_compression_gross_savings_spend: metadataOr(
      metadata.total_compression_gross_savings_spend,
      numberOf(resultTotals.compression_gross_savings_spend),
    ),
    total_compression_extra_input_spend: metadataOr(
      metadata.total_compression_extra_input_spend,
      numberOf(resultTotals.compression_extra_input_spend),
    ),
    total_prompt_caching_savings_spend: metadataOr(
      metadata.total_prompt_caching_savings_spend,
      numberOf(resultTotals.prompt_caching_savings_spend),
    ),
  };
};

const aggregateEntries = (
  entries: Array<{ id: string; metrics: SpendMetrics; metadata?: KeyMetadata }>,
): DimensionRow[] =>
  Object.values(
    entries.reduce<Record<string, DimensionRow>>((rows, entry) => {
      const previous = rows[entry.id];
      return {
        ...rows,
        [entry.id]: {
          id: entry.id,
          label: entry.metadata?.key_alias || entry.id,
          keyMetadata: entry.metadata ?? previous?.keyMetadata,
          ...addMetrics(previous ?? { id: entry.id, label: entry.id, ...emptyMetrics() }, entry.metrics),
        },
      };
    }, {}),
  ).sort((a, b) => b.total_tokens - a.total_tokens);

export const modelRowsOf = (results: DailyData[]): DimensionRow[] =>
  aggregateEntries(
    results.flatMap((day) =>
      Object.entries(day.breakdown.models).map(([id, value]) => ({ id, metrics: value.metrics })),
    ),
  );

export const apiKeyRowsOf = (results: DailyData[]): DimensionRow[] =>
  aggregateEntries(
    results.flatMap((day) =>
      Object.entries(day.breakdown.api_keys).map(([id, value]) => ({
        id,
        metrics: value.metrics,
        metadata: value.metadata,
      })),
    ),
  );

export const netRateOf = (metrics: SpendMetrics): number => {
  const net = numberOf(metrics.compression_saved_tokens);
  const baseline = metrics.prompt_tokens + net;
  return baseline > 0 ? net / baseline : 0;
};

export const shortDate = (iso: string): string =>
  new Date(`${iso}T00:00:00`).toLocaleDateString("zh-CN", { month: "short", day: "numeric" });

export const shortKey = (key: string): string => (key.length <= 22 ? key : `${key.slice(0, 12)}...${key.slice(-6)}`);

export const dailyExportRows = (response: TokenAnalyticsResponse | null) =>
  (response?.results ?? []).map((day) => ({
    日期: day.date,
    请求数: day.metrics.api_requests,
    成功请求数: day.metrics.successful_requests,
    失败请求数: day.metrics.failed_requests,
    输入Token: day.metrics.prompt_tokens,
    输出Token: day.metrics.completion_tokens,
    总Token: day.metrics.total_tokens,
    压缩次数: numberOf(day.metrics.compression_requests),
    毛节省Token: numberOf(day.metrics.compression_gross_saved_tokens),
    召回额外输入Token: numberOf(day.metrics.compression_extra_input_tokens),
    净节省Token: numberOf(day.metrics.compression_saved_tokens),
    净节省率: netRateOf(day.metrics),
    缓存读取Token: day.metrics.cache_read_input_tokens,
    压缩净节省金额USD: numberOf(day.metrics.compression_savings_spend),
    前缀缓存节省金额USD: numberOf(day.metrics.prompt_caching_savings_spend),
    模型花费USD: day.metrics.spend,
  }));
