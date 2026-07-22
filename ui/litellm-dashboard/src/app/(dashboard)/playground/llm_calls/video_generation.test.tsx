import { afterEach, describe, expect, it, vi } from "vitest";
import { makeOpenAIVideoGenerationRequest } from "./video_generation";

vi.mock("@/components/networking", () => ({ getProxyBaseUrl: () => "http://localhost:4000" }));
vi.mock("@/components/molecules/notifications_manager", () => ({
  default: { fromBackend: vi.fn() },
}));

describe("makeOpenAIVideoGenerationRequest", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("creates a video task, polls it, downloads the content, and updates the UI", async () => {
    const videoBlob = new Blob(["video"], { type: "video/mp4" });
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ id: "video-123", object: "video", status: "queued", model: "video-model" }), {
          status: 200,
        }),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify({ id: "video-123", object: "video", status: "completed", model: "video-model" }), {
          status: 200,
        }),
      )
      .mockResolvedValueOnce(new Response(videoBlob, { status: 200 }));
    const createObjectURL = vi.fn(() => "blob:video-123");
    Object.defineProperty(URL, "createObjectURL", { configurable: true, value: createObjectURL });
    const updateVideoUI = vi.fn();
    const updateTaskUI = vi.fn();

    await makeOpenAIVideoGenerationRequest({
      prompt: "A sunrise",
      updateVideoUI,
      updateTaskUI,
      selectedModel: "video-model",
      accessToken: "sk-test",
      tags: ["playground"],
      pollIntervalMs: 0,
      maxPollAttempts: 2,
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "http://localhost:4000/v1/videos",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ model: "video-model", prompt: "A sunrise" }),
      }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "http://localhost:4000/v1/videos/video-123",
      expect.objectContaining({ method: "GET" }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "http://localhost:4000/v1/videos/video-123/content",
      expect.objectContaining({ method: "GET" }),
    );
    expect(updateTaskUI).toHaveBeenCalledWith(expect.objectContaining({ id: "video-123", status: "queued" }));
    expect(createObjectURL).toHaveBeenCalledWith(videoBlob);
    expect(updateVideoUI).toHaveBeenCalledWith("blob:video-123", "video-model");
  });
});
