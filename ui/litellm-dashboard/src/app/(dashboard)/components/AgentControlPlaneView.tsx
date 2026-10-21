"use client";

import { useEffect, useRef, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { usePluginMode } from "@/contexts/PluginModeContext";
import { createApiClient } from "@/lib/http/client";
import { getProxyBaseUrl } from "@/components/networking";

const pluginApiClient = createApiClient({ getBaseUrl: () => getProxyBaseUrl() ?? "" });

export function AgentControlPlaneView() {
  const { activePlugin } = usePluginMode();
  const activePluginName = activePlugin?.name;
  const agentPlatformUrl = activePlugin?.url ?? "";
  const { accessToken } = useAuth();
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [auth, setAuth] = useState<{ plugin: string; claim: string } | null>(null);

  // Fetch a short-lived identity claim scoped to the *active* plugin. The claim
  // is encrypted under that plugin's own per-plugin key, so it must be requested
  // per plugin and re-fetched when the user switches plugins.
  useEffect(() => {
    if (!accessToken || !activePluginName) return;
    let cancelled = false;
    pluginApiClient
      .get("/api/plugins/auth-token", { accessToken, query: { plugin_name: activePluginName } })
      .then((data: { session_claim?: string }) => {
        if (!cancelled && data?.session_claim) setAuth({ plugin: activePluginName, claim: data.session_claim });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [accessToken, activePluginName]);

  // Deliver the claim to the iframe via postMessage, but only while it was issued
  // for the plugin currently mounted — never replay one plugin's claim to another.
  // targetOrigin is the configured plugin URL — no other origin receives it.
  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe || !auth || auth.plugin !== activePluginName || !agentPlatformUrl) return;
    const send = () => {
      iframe.contentWindow?.postMessage({ type: "litellm-auth", session_claim: auth.claim }, agentPlatformUrl);
    };
    // Cover both orderings: the iframe may have already fired `load` before the
    // claim arrived (send now), or it may load/reload later (send on the event).
    send();
    iframe.addEventListener("load", send);
    return () => iframe.removeEventListener("load", send);
  }, [auth, activePluginName, agentPlatformUrl]);

  if (!agentPlatformUrl) {
    return (
      <div className="flex flex-1 items-center justify-center text-gray-500">
        <div className="text-center">
          <p className="text-lg font-medium mb-2">Plugin</p>
          <p className="text-sm">Configure the plugin URL in settings</p>
        </div>
      </div>
    );
  }

  // Embed the plugin at its root; the plugin renders its own full UI (incl. nav) inside.
  return (
    <iframe
      ref={iframeRef}
      src={`${agentPlatformUrl.replace(/\/$/, "")}/`}
      style={{
        width: "100%",
        height: "100%",
        border: "none",
        flex: 1,
        minHeight: "calc(100vh - 56px)",
      }}
      title={activePlugin?.display_name ?? "Plugin"}
      allow="clipboard-write"
    />
  );
}
