"use client";

import Papa from "papaparse";
import { DatePicker } from "antd";
import zhCN from "antd/locale/zh_CN";
import dayjs from "dayjs";
import {
  Activity,
  CalendarDays,
  Database,
  Download,
  MessagesSquare,
  RefreshCw,
  Search,
  Sparkles,
  Wallet,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { sessionUsageAnalyticsCall } from "@/components/networking";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { formatNumberWithCommas } from "@/utils/dataUtils";

import {
  compressionNetRate,
  emptySessionMetrics,
  sessionExportRows,
  type SessionAnalyticsResponse,
  type SessionUsageMetrics,
  type SessionUsageSummary,
} from "./sessionAnalytics";

interface SessionAnalyticsViewProps {
  accessToken: string | null;
  userId: string | null;
}

type DateRange = { from?: Date; to?: Date };

const defaultStart = (): Date => {
  const value = new Date();
  value.setDate(value.getDate() - 30);
  value.setHours(0, 0, 0, 0);
  return value;
};

const token = (value: number): string => formatNumberWithCommas(value);
const percent = (value: number): string => `${(value * 100).toFixed(2)}%`;
const usd = (value: number): string => `$${formatNumberWithCommas(value, value !== 0 && Math.abs(value) < 1 ? 4 : 2)}`;
const timestamp = (value: string): string => new Date(value).toLocaleString("zh-CN", { hour12: false });

const MetricCard = ({
  label,
  value,
  hint,
  negative = false,
}: {
  label: string;
  value: string;
  hint: string;
  negative?: boolean;
}) => (
  <Card size="sm">
    <CardHeader>
      <CardTitle className="text-sm text-muted-foreground">{label}</CardTitle>
    </CardHeader>
    <CardContent>
      <p className={`text-2xl font-semibold tabular-nums ${negative ? "text-red-600" : "text-foreground"}`}>{value}</p>
      <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
    </CardContent>
  </Card>
);

const MetricsGrid = ({ metrics, sessionCount }: { metrics: SessionUsageMetrics; sessionCount?: number }) => (
  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
    {sessionCount !== undefined && (
      <MetricCard label="会话数" value={token(sessionCount)} hint="由 SESSION_ID 请求头识别" />
    )}
    <MetricCard
      label="请求数"
      value={token(metrics.api_requests)}
      hint={`${token(metrics.successful_requests)} 次成功 · ${token(metrics.failed_requests)} 次失败`}
    />
    <MetricCard
      label="实际消耗 Token"
      value={token(metrics.total_tokens)}
      hint={`${token(metrics.prompt_tokens)} 输入 · ${token(metrics.completion_tokens)} 输出`}
    />
    <MetricCard label="模型花费" value={usd(metrics.spend)} hint="按请求日志中的实际费用汇总" />
    <MetricCard
      label="缓存读取 Token"
      value={token(metrics.cache_read_input_tokens)}
      hint={`${token(metrics.cache_creation_input_tokens)} 缓存写入 Token`}
    />
    <MetricCard
      label="压缩次数"
      value={token(metrics.compression_requests)}
      hint={`${token(metrics.compression_gross_saved_tokens)} 毛节省 Token`}
    />
    <MetricCard
      label="净节省 Token"
      value={token(metrics.compression_saved_tokens)}
      hint={`净节省率 ${percent(compressionNetRate(metrics))}`}
      negative={metrics.compression_saved_tokens < 0}
    />
  </div>
);

const SessionModels = ({ models }: { models: string[] }) => (
  <div className="flex max-w-64 flex-wrap gap-1">
    {models.slice(0, 2).map((model) => (
      <Badge key={model} variant="secondary" className="max-w-40 truncate" title={model}>
        {model}
      </Badge>
    ))}
    {models.length > 2 && <Badge variant="outline">+{models.length - 2}</Badge>}
  </div>
);

const SessionTable = ({
  sessions,
  selectedSessionId,
  onSelect,
}: {
  sessions: SessionUsageSummary[];
  selectedSessionId: string | null;
  onSelect: (sessionId: string) => void;
}) => (
  <div className="overflow-x-auto">
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>会话 ID</TableHead>
          <TableHead>模型</TableHead>
          <TableHead>最近请求</TableHead>
          <TableHead className="text-right">请求数</TableHead>
          <TableHead className="text-right">输入 Token</TableHead>
          <TableHead className="text-right">输出 Token</TableHead>
          <TableHead className="text-right">缓存读取</TableHead>
          <TableHead className="text-right">压缩次数</TableHead>
          <TableHead className="text-right">净节省</TableHead>
          <TableHead className="text-right">总 Token</TableHead>
          <TableHead className="text-right">花费</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {sessions.length === 0 ? (
          <TableRow>
            <TableCell colSpan={11} className="h-32 text-center text-muted-foreground">
              当前筛选范围内暂无带 SESSION_ID 请求头的会话
            </TableCell>
          </TableRow>
        ) : (
          sessions.map((session) => (
            <TableRow
              key={session.session_id}
              data-state={selectedSessionId === session.session_id ? "selected" : undefined}
            >
              <TableCell>
                <button
                  type="button"
                  className="max-w-64 truncate text-left font-mono text-sm font-medium text-blue-600 hover:underline focus-visible:rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  title={session.session_id}
                  aria-label={`查看会话 ${session.session_id}`}
                  onClick={() => onSelect(session.session_id)}
                >
                  {session.session_id}
                </button>
              </TableCell>
              <TableCell>
                <SessionModels models={session.models} />
              </TableCell>
              <TableCell className="whitespace-nowrap text-sm text-muted-foreground">
                {timestamp(session.last_activity)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{token(session.api_requests)}</TableCell>
              <TableCell className="text-right tabular-nums">{token(session.prompt_tokens)}</TableCell>
              <TableCell className="text-right tabular-nums">{token(session.completion_tokens)}</TableCell>
              <TableCell className="text-right tabular-nums">{token(session.cache_read_input_tokens)}</TableCell>
              <TableCell className="text-right tabular-nums">{token(session.compression_requests)}</TableCell>
              <TableCell
                className={`text-right tabular-nums ${session.compression_saved_tokens < 0 ? "text-red-600" : ""}`}
              >
                {token(session.compression_saved_tokens)}
              </TableCell>
              <TableCell className="text-right tabular-nums">{token(session.total_tokens)}</TableCell>
              <TableCell className="text-right tabular-nums">{usd(session.spend)}</TableCell>
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  </div>
);

const SessionPageHeader = ({
  loading,
  canExport,
  onRefresh,
  onExport,
}: {
  loading: boolean;
  canExport: boolean;
  onRefresh: () => void;
  onExport: () => void;
}) => (
  <div className="flex flex-wrap items-start justify-between gap-4">
    <div>
      <div className="flex items-center gap-2">
        <MessagesSquare className="size-6 text-violet-600" strokeWidth={1.75} />
        <h1 className="text-xl font-semibold text-foreground">会话计费</h1>
      </div>
      <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
        按请求头中的 SESSION_ID 聚合 Agent 会话，统一查看 Token、缓存、压缩次数和实际花费。请求头名称不区分大小写。
      </p>
    </div>
    <div className="flex items-center gap-2">
      <Button variant="outline" onClick={onRefresh} disabled={loading}>
        <RefreshCw /> 刷新
      </Button>
      <Button variant="outline" onClick={onExport} disabled={!canExport}>
        <Download /> 导出当前页
      </Button>
    </div>
  </div>
);

const SessionFilters = ({
  dateRange,
  draftSessionId,
  draftApiKey,
  draftModel,
  filtersDirty,
  hasFilterValue,
  onDateChange,
  onSessionIdChange,
  onApiKeyChange,
  onModelChange,
  onApply,
  onClear,
}: {
  dateRange: DateRange;
  draftSessionId: string;
  draftApiKey: string;
  draftModel: string;
  filtersDirty: boolean;
  hasFilterValue: boolean;
  onDateChange: (from: Date, to: Date) => void;
  onSessionIdChange: (value: string) => void;
  onApiKeyChange: (value: string) => void;
  onModelChange: (value: string) => void;
  onApply: () => void;
  onClear: () => void;
}) => (
  <Card>
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <CalendarDays className="size-4" /> 筛选条件
      </CardTitle>
      <CardDescription>会话 ID 支持模糊搜索；API Key 与模型使用精确匹配。</CardDescription>
    </CardHeader>
    <CardContent>
      <form
        className="flex flex-wrap items-end gap-4"
        onSubmit={(event) => {
          event.preventDefault();
          onApply();
        }}
      >
        <label className="space-y-1 text-sm">
          <span className="block font-medium">时间范围</span>
          <DatePicker.RangePicker
            locale={zhCN.DatePicker}
            value={[dateRange.from ? dayjs(dateRange.from) : null, dateRange.to ? dayjs(dateRange.to) : null]}
            format="YYYY-MM-DD"
            allowClear={false}
            presets={[
              { label: "今天", value: [dayjs().startOf("day"), dayjs().endOf("day")] },
              { label: "最近 7 天", value: [dayjs().subtract(6, "day").startOf("day"), dayjs().endOf("day")] },
              { label: "最近 30 天", value: [dayjs().subtract(29, "day").startOf("day"), dayjs().endOf("day")] },
              { label: "本月", value: [dayjs().startOf("month"), dayjs().endOf("day")] },
            ]}
            onChange={(value) => {
              if (!value?.[0] || !value[1]) return;
              onDateChange(value[0].startOf("day").toDate(), value[1].endOf("day").toDate());
            }}
          />
        </label>
        <label className="min-w-60 flex-1 space-y-1 text-sm">
          <span className="font-medium">会话 ID</span>
          <div className="relative">
            <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted-foreground" />
            <input
              aria-label="会话 ID"
              className="block h-9 w-full rounded-md border border-input bg-background pl-9 pr-3 text-sm"
              placeholder="输入 SESSION_ID"
              value={draftSessionId}
              onChange={(event) => onSessionIdChange(event.target.value)}
            />
          </div>
        </label>
        <label className="min-w-52 space-y-1 text-sm">
          <span className="font-medium">API Key</span>
          <input
            aria-label="API Key"
            className="block h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            placeholder="全部 API Key"
            value={draftApiKey}
            onChange={(event) => onApiKeyChange(event.target.value)}
          />
        </label>
        <label className="min-w-52 space-y-1 text-sm">
          <span className="font-medium">模型</span>
          <input
            aria-label="模型"
            className="block h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            placeholder="全部模型"
            value={draftModel}
            onChange={(event) => onModelChange(event.target.value)}
          />
        </label>
        <Button type="submit" disabled={!filtersDirty}>
          应用筛选
        </Button>
        {hasFilterValue && (
          <Button type="button" variant="ghost" onClick={onClear}>
            清除筛选
          </Button>
        )}
      </form>
    </CardContent>
  </Card>
);

const SessionResults = ({
  error,
  loading,
  data,
  totals,
  selectedSession,
  selectedSessionId,
  page,
  onSelect,
  onPreviousPage,
  onNextPage,
}: {
  error: string;
  loading: boolean;
  data: SessionAnalyticsResponse | null;
  totals: SessionUsageMetrics;
  selectedSession: SessionUsageSummary | null;
  selectedSessionId: string | null;
  page: number;
  onSelect: (sessionId: string) => void;
  onPreviousPage: () => void;
  onNextPage: () => void;
}) => (
  <>
    {error && (
      <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        {error}
      </div>
    )}
    <section className="space-y-3">
      <div className="flex items-center gap-2">
        <Activity className="size-4 text-blue-600" />
        <h2 className="font-semibold">会话总览</h2>
      </div>
      {loading && !data ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, index) => (
            <Skeleton key={index} className="h-28" />
          ))}
        </div>
      ) : (
        <MetricsGrid metrics={totals} sessionCount={data?.total_sessions ?? 0} />
      )}
    </section>
    {selectedSession && (
      <Card className="border-violet-200 bg-violet-50/30 dark:border-violet-900 dark:bg-violet-950/10">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wallet className="size-4 text-violet-600" /> 会话详情
          </CardTitle>
          <CardDescription className="break-all font-mono">{selectedSession.session_id}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
            <span>{timestamp(selectedSession.first_activity)} 首次请求</span>
            <span>·</span>
            <span>{timestamp(selectedSession.last_activity)} 最近请求</span>
            <span>·</span>
            <SessionModels models={selectedSession.models} />
          </div>
          <MetricsGrid metrics={selectedSession} />
        </CardContent>
      </Card>
    )}
    <Card>
      <CardHeader className="flex-row items-center justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Database className="size-4" /> 会话明细
          </CardTitle>
          <CardDescription className="mt-1">点击会话 ID 查看完整指标；每页最多显示 50 个会话。</CardDescription>
        </div>
        {loading && data && <span className="text-xs text-muted-foreground">正在更新...</span>}
      </CardHeader>
      <CardContent className="space-y-4">
        <SessionTable sessions={data?.sessions ?? []} selectedSessionId={selectedSessionId} onSelect={onSelect} />
        <div className="flex flex-wrap items-center justify-between gap-3 border-t pt-4 text-sm text-muted-foreground">
          <span>
            第 {data?.page ?? page} / {Math.max(data?.total_pages ?? 0, 1)} 页，共 {token(data?.total_sessions ?? 0)}{" "}
            个会话
          </span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={loading || page <= 1} onClick={onPreviousPage}>
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={loading || !data || page >= data.total_pages}
              onClick={onNextPage}
            >
              下一页
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  </>
);

export default function SessionAnalyticsView({ accessToken, userId }: SessionAnalyticsViewProps) {
  const [dateRange, setDateRange] = useState<DateRange>({ from: defaultStart(), to: new Date() });
  const [sessionId, setSessionId] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [draftSessionId, setDraftSessionId] = useState("");
  const [draftApiKey, setDraftApiKey] = useState("");
  const [draftModel, setDraftModel] = useState("");
  const [data, setData] = useState<SessionAnalyticsResponse | null>(null);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);
  const start = dateRange.from;
  const end = dateRange.to;

  useEffect(() => {
    if (!accessToken || !start || !end) return;
    let active = true;
    const requestFilters = {
      userId,
      sessionId,
      apiKey,
      model,
      page,
      pageSize: 50,
    };
    sessionUsageAnalyticsCall(accessToken, start, end, requestFilters)
      .then((response) => {
        if (!active) return;
        setData(response as SessionAnalyticsResponse);
        setError("");
        setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setError("会话计费数据加载失败，请检查时间范围后重试。");
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [accessToken, apiKey, end, model, page, reload, sessionId, start, userId]);

  const selectedSession = useMemo(
    () => data?.sessions.find((session) => session.session_id === selectedSessionId) ?? null,
    [data, selectedSessionId],
  );
  const draftFilterSignature = [draftSessionId.trim(), draftApiKey.trim(), draftModel.trim()].join("\u0000");
  const appliedFilterSignature = [sessionId, apiKey, model].join("\u0000");
  const filtersDirty = draftFilterSignature !== appliedFilterSignature;
  const hasFilterValue = [draftSessionId, draftApiKey, draftModel, sessionId, apiKey, model].some(Boolean);
  const totals = data?.totals ?? emptySessionMetrics();

  const applyFilters = () => {
    setLoading(true);
    setPage(1);
    setSelectedSessionId(null);
    setSessionId(draftSessionId.trim());
    setApiKey(draftApiKey.trim());
    setModel(draftModel.trim());
  };

  const clearFilters = () => {
    setLoading(true);
    setPage(1);
    setSelectedSessionId(null);
    setDraftSessionId("");
    setDraftApiKey("");
    setDraftModel("");
    setSessionId("");
    setApiKey("");
    setModel("");
  };

  const exportCsv = () => {
    const csv = Papa.unparse(sessionExportRows(data?.sessions ?? []));
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `session-analytics-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const refresh = () => {
    setLoading(true);
    setError("");
    setReload((value) => value + 1);
  };

  const changeDateRange = (from: Date, to: Date) => {
    setLoading(true);
    setPage(1);
    setSelectedSessionId(null);
    setDateRange({ from, to });
  };

  const previousPage = () => {
    setLoading(true);
    setSelectedSessionId(null);
    setPage((value) => Math.max(1, value - 1));
  };

  const nextPage = () => {
    setLoading(true);
    setSelectedSessionId(null);
    setPage((value) => value + 1);
  };

  if (!accessToken) {
    return <div className="p-6 text-sm text-muted-foreground">正在等待授权...</div>;
  }

  return (
    <div className="w-full space-y-6 p-6">
      <SessionPageHeader
        loading={loading}
        canExport={Boolean(data?.sessions.length)}
        onRefresh={refresh}
        onExport={exportCsv}
      />
      <SessionFilters
        dateRange={dateRange}
        draftSessionId={draftSessionId}
        draftApiKey={draftApiKey}
        draftModel={draftModel}
        filtersDirty={filtersDirty}
        hasFilterValue={hasFilterValue}
        onDateChange={changeDateRange}
        onSessionIdChange={setDraftSessionId}
        onApiKeyChange={setDraftApiKey}
        onModelChange={setDraftModel}
        onApply={applyFilters}
        onClear={clearFilters}
      />
      <SessionResults
        error={error}
        loading={loading}
        data={data}
        totals={totals}
        selectedSession={selectedSession}
        selectedSessionId={selectedSessionId}
        page={page}
        onSelect={setSelectedSessionId}
        onPreviousPage={previousPage}
        onNextPage={nextPage}
      />

      <Card size="sm">
        <CardContent className="flex items-start gap-3 pt-4 text-sm text-muted-foreground">
          <Sparkles className="mt-0.5 size-4 shrink-0 text-violet-600" />
          <p>
            Agent 应用只需在每次模型请求中携带同一个 SESSION_ID
            请求头。系统按会话聚合统计；未携带该请求头的普通请求不会出现在这里。
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
