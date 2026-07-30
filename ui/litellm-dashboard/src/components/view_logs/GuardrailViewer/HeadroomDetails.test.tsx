import React from "react";
import { describe, it, expect } from "vitest";
import userEvent from "@testing-library/user-event";
import { renderWithProviders, screen } from "../../../../tests/test-utils";
import HeadroomDetails from "./HeadroomDetails";

describe("HeadroomDetails", () => {
  it("renders compression stats and before/after messages", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <HeadroomDetails
        response={{
          tokens_before: 1000,
          tokens_after: 400,
          tokens_saved: 600,
          compression_ratio: 0.4,
          transforms_applied: ["router:smart_crusher:0.35"],
          messages_before: [{ role: "user", content: "long prompt" }],
          messages_after: [{ role: "user", content: "short" }],
        }}
      />,
    );

    expect(screen.getByText("Tokens before")).toBeInTheDocument();
    expect(screen.getByText("1000")).toBeInTheDocument();
    expect(screen.getByText("400")).toBeInTheDocument();
    expect(screen.getByText("600")).toBeInTheDocument();
    expect(screen.getByText("0.40")).toBeInTheDocument();
    expect(screen.getByText("router:smart_crusher:0.35")).toBeInTheDocument();

    expect(screen.getByText(/long prompt/)).toBeInTheDocument();
    expect(screen.getByText(/"content": "short"/)).toBeInTheDocument();

    await user.click(screen.getByText("Messages before compression"));
    expect(screen.queryByText(/long prompt/)).not.toBeInTheDocument();
  });

  it("shows empty-state copy when message arrays are missing", () => {
    renderWithProviders(
      <HeadroomDetails
        response={{
          tokens_before: 10,
          tokens_after: 5,
          tokens_saved: 5,
          compression_ratio: 0.5,
        }}
      />,
    );

    expect(screen.getByText("Messages before compression were not stored for this request.")).toBeInTheDocument();
    expect(screen.getByText("Messages after compression were not stored for this request.")).toBeInTheDocument();
  });

  it("renders completed CCR telemetry", () => {
    renderWithProviders(
      <HeadroomDetails
        response={{
          ccr_enabled: true,
          ccr_status: "completed",
          ccr_hashes_issued: 2,
          ccr_hashes_requested: 1,
          ccr_hashes_retrieved: 1,
          ccr_retrieved_chars: 4096,
          ccr_followup_model: "openai/followup-model",
          ccr_fallback_used: true,
        }}
      />,
    );

    expect(screen.getByText("COMPLETED")).toBeInTheDocument();
    expect(screen.getByText("openai/followup-model")).toBeInTheDocument();
    expect(screen.getByText("Hashes issued")).toBeInTheDocument();
    expect(screen.getByText("4096")).toBeInTheDocument();
    expect(screen.getByText("Yes")).toBeInTheDocument();
  });

  it("renders follow-up failure details", () => {
    renderWithProviders(
      <HeadroomDetails
        response={{
          ccr_status: "followup_failed",
          ccr_hashes_issued: 1,
          ccr_hashes_requested: 1,
          ccr_hashes_retrieved: 1,
          ccr_retrieved_chars: 512,
          ccr_fallback_used: false,
          ccr_error: "followup_failed",
        }}
      />,
    );

    expect(screen.getByText("FOLLOWUP FAILED")).toBeInTheDocument();
    expect(screen.getByText("Error: followup_failed")).toBeInTheDocument();
    expect(screen.getByText("No")).toBeInTheDocument();
  });

  it("returns null for empty object responses", () => {
    const { container } = renderWithProviders(<HeadroomDetails response={{}} />);
    expect(container.firstChild).toBeNull();
  });
});
