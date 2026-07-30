import React, { useState } from "react";

export interface HeadroomCompressionResponse {
  tokens_before?: number;
  tokens_after?: number;
  tokens_saved?: number;
  compression_ratio?: number;
  transforms_applied?: string[];
  messages_before?: unknown[];
  messages_after?: unknown[];
  ccr_enabled?: boolean;
  ccr_status?: string;
  ccr_hashes_issued?: number;
  ccr_hashes_requested?: number;
  ccr_hashes_retrieved?: number;
  ccr_retrieved_chars?: number;
  ccr_followup_model?: string;
  ccr_fallback_used?: boolean;
  ccr_error?: string | null;
}

interface HeadroomDetailsProps {
  response: HeadroomCompressionResponse | string | null | undefined;
}

const Section: React.FC<{
  title: string;
  defaultOpen?: boolean;
  children?: React.ReactNode;
}> = ({ title, defaultOpen = false, children }) => {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="border rounded-lg overflow-hidden">
      <div
        className="flex items-center justify-between p-3 bg-gray-50 cursor-pointer hover:bg-gray-100"
        onClick={() => setOpen((v) => !v)}
      >
        <div className="flex items-center">
          <svg
            className={`w-5 h-5 mr-2 transition-transform ${open ? "transform rotate-90" : ""}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
          <h5 className="font-medium text-sm">{title}</h5>
        </div>
      </div>
      {open && <div className="p-3 border-t bg-white">{children}</div>}
    </div>
  );
};

const formatNumber = (value: number | undefined): string => {
  if (value == null || Number.isNaN(value)) return "-";
  return Number.isInteger(value) ? String(value) : value.toFixed(2);
};

const ccrStatusStyle = (status: string): string => {
  if (status === "completed") return "border-green-200 bg-green-50 text-green-700";
  if (status === "retrieve_success") return "border-blue-200 bg-blue-50 text-blue-700";
  if (status === "retrieve_failed" || status === "followup_failed") {
    return "border-red-200 bg-red-50 text-red-700";
  }
  return "border-gray-200 bg-gray-50 text-gray-700";
};

const formatStatus = (status: string): string => status.replace(/_/g, " ").toUpperCase();

const hasDefinedValue = (...values: unknown[]): boolean => values.some((value) => value != null);

const hasItems = (values: unknown[] | undefined): boolean => Boolean(values?.length);

const MessagesBlock: React.FC<{ messages: unknown[] | undefined; emptyLabel: string }> = ({ messages, emptyLabel }) => {
  if (!messages || messages.length === 0) {
    return <p className="text-sm text-gray-500">{emptyLabel}</p>;
  }
  return (
    <pre className="bg-gray-50 rounded-sm p-3 text-xs overflow-x-auto whitespace-pre-wrap break-words">
      {JSON.stringify(messages, null, 2)}
    </pre>
  );
};

const HeadroomDetails: React.FC<HeadroomDetailsProps> = ({ response }) => {
  if (typeof response === "string") {
    return response ? (
      <div className="mt-3">
        <p className="text-sm text-gray-600">{response}</p>
      </div>
    ) : null;
  }
  if (!response) return null;

  const hasStats = hasDefinedValue(
    response.tokens_before,
    response.tokens_after,
    response.tokens_saved,
    response.compression_ratio,
  );
  const hasMessages = hasItems(response.messages_before) || hasItems(response.messages_after);
  const hasCcrTelemetry = Boolean(response.ccr_status);
  const hasTransforms = hasItems(response.transforms_applied);
  const hasDetails = [hasStats, hasMessages, hasCcrTelemetry, hasTransforms].some(Boolean);

  if (!hasDetails) return null;

  return (
    <div className="mt-3 space-y-3">
      {hasCcrTelemetry && response.ccr_status && (
        <div className={`rounded-lg border p-3 ${ccrStatusStyle(response.ccr_status)}`}>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="text-[11px] uppercase tracking-wide opacity-75">CCR status</div>
              <div className="mt-1 text-sm font-semibold">{formatStatus(response.ccr_status)}</div>
            </div>
            {response.ccr_followup_model && (
              <div className="text-right">
                <div className="text-[11px] uppercase tracking-wide opacity-75">Follow-up model</div>
                <div className="mt-1 font-mono text-xs">{response.ccr_followup_model}</div>
              </div>
            )}
          </div>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
            <div>
              <div className="text-[11px] uppercase tracking-wide opacity-75">Hashes issued</div>
              <div className="mt-1 font-mono text-sm">{formatNumber(response.ccr_hashes_issued)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide opacity-75">Requested</div>
              <div className="mt-1 font-mono text-sm">{formatNumber(response.ccr_hashes_requested)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide opacity-75">Retrieved</div>
              <div className="mt-1 font-mono text-sm">{formatNumber(response.ccr_hashes_retrieved)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide opacity-75">Retrieved chars</div>
              <div className="mt-1 font-mono text-sm">{formatNumber(response.ccr_retrieved_chars)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wide opacity-75">Fallback used</div>
              <div className="mt-1 font-mono text-sm">{response.ccr_fallback_used ? "Yes" : "No"}</div>
            </div>
          </div>
          {response.ccr_error && <div className="mt-3 font-mono text-xs">Error: {response.ccr_error}</div>}
        </div>
      )}

      {hasStats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-lg border border-gray-200 bg-white p-3">
            <div className="text-[11px] uppercase tracking-wide text-gray-500">Tokens before</div>
            <div className="mt-1 font-mono text-sm text-gray-900">{formatNumber(response.tokens_before)}</div>
          </div>
          <div className="rounded-lg border border-gray-200 bg-white p-3">
            <div className="text-[11px] uppercase tracking-wide text-gray-500">Tokens after</div>
            <div className="mt-1 font-mono text-sm text-gray-900">{formatNumber(response.tokens_after)}</div>
          </div>
          <div className="rounded-lg border border-gray-200 bg-white p-3">
            <div className="text-[11px] uppercase tracking-wide text-gray-500">Tokens saved</div>
            <div className="mt-1 font-mono text-sm text-gray-900">{formatNumber(response.tokens_saved)}</div>
          </div>
          <div className="rounded-lg border border-gray-200 bg-white p-3">
            <div className="text-[11px] uppercase tracking-wide text-gray-500">Compression ratio</div>
            <div className="mt-1 font-mono text-sm text-gray-900">{formatNumber(response.compression_ratio)}</div>
          </div>
        </div>
      )}

      {response.transforms_applied && response.transforms_applied.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {response.transforms_applied.map((transform) => (
            <span key={transform} className="px-2 py-1 bg-slate-100 text-slate-700 rounded-sm text-xs font-medium">
              {transform}
            </span>
          ))}
        </div>
      )}

      <Section title="Messages before compression" defaultOpen>
        <MessagesBlock
          messages={response.messages_before}
          emptyLabel="Messages before compression were not stored for this request."
        />
      </Section>

      <Section title="Messages after compression" defaultOpen>
        <MessagesBlock
          messages={response.messages_after}
          emptyLabel="Messages after compression were not stored for this request."
        />
      </Section>
    </div>
  );
};

export default HeadroomDetails;
