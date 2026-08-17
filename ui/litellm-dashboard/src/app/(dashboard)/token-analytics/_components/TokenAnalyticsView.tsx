"use client";

import Papa from "papaparse";
import { DatePicker } from "antd";
import zhCN from "antd/locale/zh_CN";
import dayjs from "dayjs";
import { Activity, CalendarDays, Database, Download, Gauge, RefreshCw, Sparkles, Wallet } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";

import { AreaChart } from "@/components/shared/charts";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { userDailyActivityAggregatedFilteredCall } from "@/components/networking";
import { formatNumberWithCommas } from "@/utils/dataUtils";

import {
  apiKeyRowsOf,
  dailyExportRows,
  modelRowsOf,
  netRateOf,
  shortDate,
  shortKey,
  totalsFromResponse,
  type DimensionRow,
  type TokenAnalyticsResponse,
} from "./tokenAnalytics";

interface TokenAnalyticsViewProps {
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

const DimensionTable = ({ rows, dimension }: { rows: DimensionRow[]; dimension: "model" | "key" }) => (
  <Table>
    <TableHeader>
      <TableRow>
        <TableHead>{dimension === "model" ? "模型" : "API Key"}</TableHead>
        <TableHead className="text-right">请求数</TableHead>
        <TableHead className="text-right">压缩次数</TableHead>
        <TableHead className="text-right">毛节省</TableHead>
        <TableHead className="text-right">召回输入</TableHead>
        <TableHead className="text-right">净节省</TableHead>
        <TableHead className="text-right">净节省率</TableHead>
        <TableHead className="text-right">缓存读取</TableHead>
        <TableHead className="text-right">总 Token</TableHead>
        <TableHead className="text-right">花费</TableHead>
      </TableRow>
    </TableHeader>
    <TableBody>
      {rows.length === 0 ? (
        <TableRow>
          <TableCell colSpan={10} className="h-28 text-center text-muted-foreground">
            当前时间范围内暂无数据
          </TableCell>
        </TableRow>
      ) : (
        rows.map((row) => (
          <TableRow key={row.id}>
            <TableCell>
              <div className="max-w-64">
                <p className="truncate font-medium" title={row.label}>
                  {row.label}
                </p>
                {dimension === "key" && row.label !== row.id && (
                  <p className="truncate text-xs text-muted-foreground" title={row.id}>
                    {shortKey(row.id)}
                  </p>
                )}
              </div>
            </TableCell>
            <TableCell className="text-right tabular-nums">{token(row.api_requests)}</TableCell>
            <TableCell className="text-right tabular-nums">{token(row.compression_requests ?? 0)}</TableCell>
            <TableCell className="text-right tabular-nums">{token(row.compression_gross_saved_tokens ?? 0)}</TableCell>
            <TableCell className="text-right tabular-nums">{token(row.compression_extra_input_tokens ?? 0)}</TableCell>
            <TableCell
              className={`text-right tabular-nums ${(row.compression_saved_tokens ?? 0) < 0 ? "text-red-600" : ""}`}
            >
              {token(row.compression_saved_tokens ?? 0)}
            </TableCell>
            <TableCell className="text-right tabular-nums">{percent(netRateOf(row))}</TableCell>
            <TableCell className="text-right tabular-nums">{token(row.cache_read_input_tokens)}</TableCell>
            <TableCell className="text-right tabular-nums">{token(row.total_tokens)}</TableCell>
            <TableCell className="text-right tabular-nums">{usd(row.spend)}</TableCell>
          </TableRow>
        ))
      )}
    </TableBody>
  </Table>
);

export default function TokenAnalyticsView({ accessToken, userId }: TokenAnalyticsViewProps) {
  const [dateRange, setDateRange] = useState<DateRange>({ from: defaultStart(), to: new Date() });
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");
  const [draftApiKey, setDraftApiKey] = useState("");
  const [draftModel, setDraftModel] = useState("");
  const [data, setData] = useState<TokenAnalyticsResponse | null>(null);
  const [catalog, setCatalog] = useState<TokenAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [reload, setReload] = useState(0);

  const start = dateRange.from;
  const end = dateRange.to;

  useEffect(() => {
    if (!accessToken || !start || !end) return;
    let active = true;
    userDailyActivityAggregatedFilteredCall(accessToken, start, end, { userId })
      .then((response) => {
        if (active) setCatalog(response as TokenAnalyticsResponse);
      })
      .catch(() => {
        if (active) setCatalog(null);
      });
    return () => {
      active = false;
    };
  }, [accessToken, end, reload, start, userId]);

  useEffect(() => {
    if (!accessToken || !start || !end) return;
    let active = true;
    userDailyActivityAggregatedFilteredCall(accessToken, start, end, { userId, apiKey, model })
      .then((response) => {
        if (!active) return;
        setData(response as TokenAnalyticsResponse);
        setLoading(false);
      })
      .catch(() => {
        if (!active) return;
        setData(null);
        setError("Token 分析数据加载失败，请检查时间范围后重试。");
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [accessToken, apiKey, end, model, reload, start, userId]);

  const totals = useMemo(() => totalsFromResponse(data), [data]);
  const apiKeyRows = useMemo(() => apiKeyRowsOf(data?.results ?? []), [data]);
  const modelRows = useMemo(() => modelRowsOf(data?.results ?? []), [data]);
  const catalogKeys = useMemo(() => apiKeyRowsOf(catalog?.results ?? []), [catalog]);
  const catalogModels = useMemo(() => modelRowsOf(catalog?.results ?? []), [catalog]);
  const filtersDirty = draftApiKey.trim() !== apiKey || draftModel.trim() !== model;
  const hasFilterValue = Boolean(draftApiKey || draftModel || apiKey || model);

  const usageTrend = useMemo(
    () =>
      [...(data?.results ?? [])].reverse().map((day) => ({
        date: shortDate(day.date),
        "输入 Token": day.metrics.prompt_tokens,
        "输出 Token": day.metrics.completion_tokens,
      })),
    [data],
  );
  const compressionTrend = useMemo(
    () =>
      [...(data?.results ?? [])].reverse().map((day) => ({
        date: shortDate(day.date),
        "毛节省 Token": day.metrics.compression_gross_saved_tokens ?? 0,
        "召回额外输入 Token": day.metrics.compression_extra_input_tokens ?? 0,
        "净节省 Token": day.metrics.compression_saved_tokens ?? 0,
      })),
    [data],
  );
  const cacheTrend = useMemo(
    () =>
      [...(data?.results ?? [])].reverse().map((day) => ({
        date: shortDate(day.date),
        "缓存读取 Token": day.metrics.cache_read_input_tokens,
      })),
    [data],
  );

  const exportCsv = () => {
    const csv = Papa.unparse(dailyExportRows(data));
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `token-analytics-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const applyFilters = () => {
    setLoading(true);
    setError("");
    setApiKey(draftApiKey.trim());
    setModel(draftModel.trim());
  };

  const clearFilters = () => {
    setLoading(true);
    setError("");
    setDraftApiKey("");
    setDraftModel("");
    setApiKey("");
    setModel("");
  };

  if (!accessToken) {
    return <div className="p-6 text-sm text-muted-foreground">正在等待授权...</div>;
  }

  return (
    <div className="w-full space-y-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Gauge className="size-6 text-violet-600" strokeWidth={1.75} />
            <h1 className="text-xl font-semibold text-foreground">Token 分析</h1>
          </div>
          <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
            在一个界面查看 Token 消耗、压缩收益和模型厂商前缀缓存收益。
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => {
              setLoading(true);
              setError("");
              setReload((value) => value + 1);
            }}
            disabled={loading}
          >
            <RefreshCw /> 刷新
          </Button>
          <Button variant="outline" onClick={exportCsv} disabled={!data || data.results.length === 0}>
            <Download /> 导出 CSV
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CalendarDays className="size-4" /> 筛选条件
          </CardTitle>
          <CardDescription>当前接口按天聚合数据，因此最小精确时间单位为一个自然日。</CardDescription>
        </CardHeader>
        <CardContent>
          <form
            className="flex flex-wrap items-end gap-4"
            onSubmit={(event) => {
              event.preventDefault();
              applyFilters();
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
                  setLoading(true);
                  setError("");
                  setDateRange({ from: value[0].startOf("day").toDate(), to: value[1].endOf("day").toDate() });
                }}
              />
            </label>
            <label className="min-w-60 space-y-1 text-sm">
              <span className="font-medium">API Key</span>
              <input
                aria-label="API Key"
                list="token-analytics-api-keys"
                className="block h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                placeholder="全部或手动输入 API Key"
                value={draftApiKey}
                onChange={(event) => setDraftApiKey(event.target.value)}
              />
              <datalist id="token-analytics-api-keys">
                {catalogKeys.map((row) => (
                  <option
                    key={row.id}
                    value={row.id}
                    label={
                      row.keyMetadata?.key_alias
                        ? `${row.keyMetadata.key_alias} (${shortKey(row.id)})`
                        : shortKey(row.id)
                    }
                  />
                ))}
              </datalist>
            </label>
            <label className="min-w-60 space-y-1 text-sm">
              <span className="font-medium">模型</span>
              <input
                aria-label="模型"
                list="token-analytics-models"
                className="block h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                placeholder="全部或手动输入模型"
                value={draftModel}
                onChange={(event) => setDraftModel(event.target.value)}
              />
              <datalist id="token-analytics-models">
                {catalogModels.map((row) => (
                  <option key={row.id} value={row.id} label={row.label} />
                ))}
              </datalist>
            </label>
            <Button type="submit" disabled={!filtersDirty}>
              应用筛选
            </Button>
            {hasFilterValue && (
              <Button type="button" variant="ghost" onClick={clearFilters}>
                清除筛选
              </Button>
            )}
          </form>
        </CardContent>
      </Card>

      {error && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-4 text-sm text-red-700">
          {error}
        </div>
      )}

      {loading && !data ? (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, index) => (
            <Skeleton key={index} className="h-28" />
          ))}
        </div>
      ) : (
        <>
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Activity className="size-4 text-blue-600" />
              <h2 className="font-semibold">Token 消耗</h2>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
              <MetricCard
                label="请求数"
                value={token(totals.total_api_requests)}
                hint={`${token(totals.total_successful_requests)} 次成功 · ${token(totals.total_failed_requests)} 次失败`}
              />
              <MetricCard
                label="实际消耗 Token"
                value={token(totals.total_tokens)}
                hint="实际上游处理的输入 Token + 输出 Token"
              />
              <MetricCard
                label="输入 Token"
                value={token(totals.total_prompt_tokens)}
                hint="模型厂商返回的 Prompt 用量"
              />
              <MetricCard
                label="输出 Token"
                value={token(totals.total_completion_tokens)}
                hint={`模型调用花费 ${usd(totals.total_spend)}`}
              />
            </div>
          </section>

          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Sparkles className="size-4 text-violet-600" />
              <h2 className="font-semibold">压缩收益</h2>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-5">
              <MetricCard
                label="压缩次数"
                value={token(totals.total_compression_requests)}
                hint="毛节省 Token 大于 0 的请求数"
              />
              <MetricCard
                label="毛节省 Token"
                value={token(totals.total_compression_gross_saved_tokens)}
                hint={`压缩毛收益 ${usd(totals.total_compression_gross_savings_spend)}`}
              />
              <MetricCard
                label="召回额外输入 Token"
                value={token(totals.total_compression_extra_input_tokens)}
                hint={`召回额外成本 ${usd(totals.total_compression_extra_input_spend)}`}
              />
              <MetricCard
                label="净节省 Token"
                value={token(totals.total_compression_saved_tokens)}
                hint={`压缩净收益 ${usd(totals.total_compression_savings_spend)}`}
                negative={totals.total_compression_saved_tokens < 0}
              />
              <MetricCard
                label="净节省率"
                value={percent(totals.total_compression_net_savings_rate)}
                hint="净节省 Token ÷ 原始输入基线"
                negative={totals.total_compression_net_savings_rate < 0}
              />
            </div>
          </section>

          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <Database className="size-4 text-emerald-600" />
              <h2 className="font-semibold">模型厂商前缀缓存</h2>
            </div>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <MetricCard
                label="缓存读取 Token"
                value={token(totals.total_cache_read_input_tokens)}
                hint="按厂商缓存折扣价计费，并非完全免费的 Token"
              />
              <MetricCard
                label="缓存节省金额"
                value={usd(totals.total_prompt_caching_savings_spend)}
                hint="普通输入价格减去缓存读取价格"
              />
            </div>
          </section>
        </>
      )}

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Token 消耗趋势</CardTitle>
            <CardDescription>上游模型实际处理的输入与输出 Token</CardDescription>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={usageTrend}
              index="date"
              categories={["输入 Token", "输出 Token"]}
              colors={["blue", "violet"]}
              valueFormatter={token}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>压缩收益趋势</CardTitle>
            <CardDescription>毛节省减去召回额外输入，得到净节省</CardDescription>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={compressionTrend}
              index="date"
              categories={["毛节省 Token", "召回额外输入 Token", "净节省 Token"]}
              colors={["emerald", "orange", "violet"]}
              valueFormatter={token}
            />
          </CardContent>
        </Card>
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>前缀缓存趋势</CardTitle>
            <CardDescription>模型厂商返回的缓存读取 Token</CardDescription>
          </CardHeader>
          <CardContent>
            <AreaChart
              data={cacheTrend}
              index="date"
              categories={["缓存读取 Token"]}
              colors={["emerald"]}
              valueFormatter={token}
            />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wallet className="size-4" /> 模型明细
          </CardTitle>
          <CardDescription>按当前时间范围和筛选条件聚合</CardDescription>
        </CardHeader>
        <CardContent>
          <DimensionTable rows={modelRows} dimension="model" />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>API Key 明细</CardTitle>
          <CardDescription>优先显示 Key 别名，筛选时仍使用数据库中存储的 Key 哈希</CardDescription>
        </CardHeader>
        <CardContent>
          <DimensionTable rows={apiKeyRows} dimension="key" />
        </CardContent>
      </Card>
    </div>
  );
}
