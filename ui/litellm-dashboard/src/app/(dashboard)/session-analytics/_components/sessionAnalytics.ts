export interface SessionUsageMetrics {
  spend: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  api_requests: number;
  successful_requests: number;
  failed_requests: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
  compression_saved_tokens: number;
  compression_gross_saved_tokens: number;
  compression_extra_input_tokens: number;
  compression_requests: number;
}

export interface SessionUsageSummary extends SessionUsageMetrics {
  hermes_session_id: string;
  first_activity: string;
  last_activity: string;
  models: string[];
}

export interface SessionAnalyticsResponse {
  sessions: SessionUsageSummary[];
  totals: SessionUsageMetrics;
  total_sessions: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export const emptySessionMetrics = (): SessionUsageMetrics => ({
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
});

export const compressionNetRate = (metrics: SessionUsageMetrics): number => {
  if (metrics.total_tokens <= 0 || metrics.prompt_tokens <= 0) return 0;
  const baseline = metrics.prompt_tokens + metrics.compression_saved_tokens;
  return baseline > 0 ? metrics.compression_saved_tokens / baseline : 0;
};

export const sessionExportRows = (sessions: SessionUsageSummary[]) =>
  sessions.map((session) => ({
    Hermes会话ID: session.hermes_session_id,
    首次请求: session.first_activity,
    最近请求: session.last_activity,
    模型: session.models.join(", "),
    请求数: session.api_requests,
    成功请求数: session.successful_requests,
    失败请求数: session.failed_requests,
    输入Token: session.prompt_tokens,
    输出Token: session.completion_tokens,
    总Token: session.total_tokens,
    缓存读取Token: session.cache_read_input_tokens,
    缓存写入Token: session.cache_creation_input_tokens,
    压缩次数: session.compression_requests,
    毛节省Token: session.compression_gross_saved_tokens,
    额外输入Token: session.compression_extra_input_tokens,
    净节省Token: session.compression_saved_tokens,
    花费USD: session.spend,
  }));
