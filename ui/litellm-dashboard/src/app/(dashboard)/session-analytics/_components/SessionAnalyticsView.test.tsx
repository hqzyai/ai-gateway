import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { emptySessionMetrics, type SessionAnalyticsResponse } from "./sessionAnalytics";

const analyticsCall = vi.fn();

vi.mock("@/components/networking", () => ({
  sessionUsageAnalyticsCall: (...args: unknown[]) => analyticsCall(...args),
}));

const response = (): SessionAnalyticsResponse => ({
  sessions: [
    {
      ...emptySessionMetrics(),
      session_id: "agent-session-42",
      first_activity: "2026-08-17T08:00:00Z",
      last_activity: "2026-08-17T09:00:00Z",
      models: ["gpt-4o"],
      spend: 0.12,
      prompt_tokens: 1000,
      completion_tokens: 200,
      total_tokens: 1200,
      api_requests: 3,
      successful_requests: 2,
      failed_requests: 1,
      cache_read_input_tokens: 400,
      compression_requests: 2,
      compression_gross_saved_tokens: 350,
      compression_extra_input_tokens: 50,
      compression_saved_tokens: 300,
    },
  ],
  totals: {
    ...emptySessionMetrics(),
    spend: 0.12,
    prompt_tokens: 1000,
    completion_tokens: 200,
    total_tokens: 1200,
    api_requests: 3,
    successful_requests: 2,
    failed_requests: 1,
    cache_read_input_tokens: 400,
    compression_requests: 2,
    compression_gross_saved_tokens: 350,
    compression_extra_input_tokens: 50,
    compression_saved_tokens: 300,
  },
  total_sessions: 1,
  page: 1,
  page_size: 50,
  total_pages: 1,
});

import SessionAnalyticsView from "./SessionAnalyticsView";

describe("SessionAnalyticsView", () => {
  beforeEach(() => {
    analyticsCall.mockReset();
    analyticsCall.mockResolvedValue(response());
  });

  it("shows session billing totals and expands a selected session", async () => {
    render(<SessionAnalyticsView accessToken="token" userId={null} />);

    await waitFor(() => expect(screen.getByRole("button", { name: "查看会话 agent-session-42" })).toBeInTheDocument());
    expect(screen.getByText("会话计费")).toBeInTheDocument();
    expect(screen.getAllByText("缓存读取 Token")).not.toHaveLength(0);
    expect(screen.getAllByText("压缩次数")).not.toHaveLength(0);
    fireEvent.click(screen.getByRole("button", { name: "查看会话 agent-session-42" }));
    expect(screen.getByText("会话详情")).toBeInTheDocument();
    expect(screen.getAllByText("净节省率 23.08%")).not.toHaveLength(0);
  });

  it("submits session, API key, and model filters together", async () => {
    render(<SessionAnalyticsView accessToken="token" userId="user-1" />);
    await waitFor(() => expect(analyticsCall).toHaveBeenCalledTimes(1));
    analyticsCall.mockClear();

    fireEvent.change(screen.getByLabelText("会话 ID"), { target: { value: "session-42" } });
    fireEvent.change(screen.getByLabelText("API Key"), { target: { value: "key-hash" } });
    fireEvent.change(screen.getByLabelText("模型"), { target: { value: "gpt-4o" } });
    fireEvent.click(screen.getByRole("button", { name: "应用筛选" }));

    const expectedFilters = {
      userId: "user-1",
      sessionId: "session-42",
      apiKey: "key-hash",
      model: "gpt-4o",
      page: 1,
      pageSize: 50,
    };

    await waitFor(() =>
      expect(analyticsCall).toHaveBeenCalledWith("token", expect.any(Date), expect.any(Date), expectedFilters),
    );
  });
});
