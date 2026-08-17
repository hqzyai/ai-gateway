import { describe, expect, it } from "vitest";

import {
  compressionNetRate,
  emptySessionMetrics,
  sessionExportRows,
  type SessionUsageSummary,
} from "./sessionAnalytics";

const session = (overrides: Partial<SessionUsageSummary> = {}): SessionUsageSummary => ({
  ...emptySessionMetrics(),
  session_id: "agent-session-42",
  first_activity: "2026-08-17T08:00:00Z",
  last_activity: "2026-08-17T09:00:00Z",
  models: ["gpt-4o"],
  ...overrides,
});

describe("session analytics calculations", () => {
  it("calculates a signed compression rate from the original input baseline", () => {
    expect(compressionNetRate(session({ prompt_tokens: 1250, compression_saved_tokens: -250 }))).toBe(-0.25);
  });

  it("exports cache and compression metrics as separate billing columns", () => {
    const exportMetrics = {
      cache_read_input_tokens: 400,
      cache_creation_input_tokens: 50,
      compression_requests: 2,
      compression_gross_saved_tokens: 350,
      compression_extra_input_tokens: 50,
      compression_saved_tokens: 300,
    };
    const expectedRow = {
      会话ID: "agent-session-42",
      缓存读取Token: 400,
      缓存写入Token: 50,
      压缩次数: 2,
      毛节省Token: 350,
      额外输入Token: 50,
      净节省Token: 300,
    };
    const rows = sessionExportRows([session(exportMetrics)]);

    expect(rows[0]).toMatchObject(expectedRow);
  });
});
