import { getProxyBaseUrl } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import { createApiClient } from "@/lib/http/client";

export interface VideoTask {
  id: string;
  object: "video";
  status: string;
  model?: string;
  error?: Record<string, unknown> | null;
}

const waitForNextPoll = (pollIntervalMs: number, signal?: AbortSignal): Promise<void> =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(signal.reason);
      return;
    }

    const onAbort = () => {
      clearTimeout(timeoutId);
      reject(signal?.reason);
    };
    const timeoutId = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, pollIntervalMs);
    signal?.addEventListener("abort", onAbort, { once: true });
  });

interface VideoGenerationRequestOptions {
  prompt: string;
  updateVideoUI: (videoUrl: string, model: string) => void;
  updateTaskUI: (task: VideoTask) => void;
  selectedModel: string;
  accessToken: string;
  tags?: string[];
  signal?: AbortSignal;
  customBaseUrl?: string;
  pollIntervalMs?: number;
  maxPollAttempts?: number;
}

export async function makeOpenAIVideoGenerationRequest({
  prompt,
  updateVideoUI,
  updateTaskUI,
  selectedModel,
  accessToken,
  tags,
  signal,
  customBaseUrl,
  pollIntervalMs = 3000,
  maxPollAttempts = 300,
}: VideoGenerationRequestOptions): Promise<void> {
  const proxyBaseUrl = (customBaseUrl || getProxyBaseUrl()).replace(/\/$/, "");
  const client = createApiClient({ getBaseUrl: () => proxyBaseUrl });
  const headers = tags && tags.length > 0 ? { "x-litellm-tags": tags.join(",") } : undefined;
  const requestOptions = { accessToken, headers, signal };

  try {
    const createdTask = await client.post<VideoTask>("/v1/videos", {
      ...requestOptions,
      body: { model: selectedModel, prompt },
    });
    updateTaskUI(createdTask);

    const completedTask = await Array.from({ length: maxPollAttempts }).reduce<Promise<VideoTask>>(
      async (taskPromise) => {
        const task = await taskPromise;
        if (task.status === "completed") {
          return task;
        }
        if (["failed", "cancelled", "expired"].includes(task.status)) {
          throw new Error(JSON.stringify(task.error ?? { status: task.status }));
        }

        await waitForNextPoll(pollIntervalMs, signal);
        return client.get<VideoTask>(`/v1/videos/${encodeURIComponent(task.id)}`, requestOptions);
      },
      Promise.resolve(createdTask),
    );

    if (completedTask.status !== "completed") {
      throw new Error("Video generation timed out before completion");
    }

    const videoBlob = await client.getBlob(`/v1/videos/${encodeURIComponent(completedTask.id)}/content`, {
      accessToken,
      signal,
    });
    const videoUrl = URL.createObjectURL(videoBlob);
    updateVideoUI(videoUrl, completedTask.model || selectedModel);
  } catch (error) {
    if (!signal?.aborted) {
      NotificationManager.fromBackend(`Error occurred while generating video. Error: ${error}`);
    }
    throw error;
  }
}
