"use client";

import React, { Suspense, useState, useEffect } from "react";
import { DashboardHeader } from "@/components/DashboardHeader";
import Navbar from "@/components/navbar";
import LoadingScreen from "@/components/common_components/LoadingScreen";
import { ThemeProvider } from "@/contexts/ThemeContext";
import { useAuth } from "@/contexts/AuthContext";
import SidebarProvider from "@/app/(dashboard)/components/SidebarProvider";
import { useRouter, useSearchParams, usePathname } from "next/navigation";
import { DebugWarningBanner } from "@/components/DebugWarningBanner";
import { LicenseExpiryBanner } from "@/components/LicenseExpiryBanner";
import { MIGRATED_PAGES, migratedHref, legacyPageHref, legacyKeyForPathname } from "@/utils/migratedPages";
import { PluginModeProvider, usePluginMode } from "@/contexts/PluginModeContext";
import { AgentControlPlaneView } from "./components/AgentControlPlaneView";

// Wrapper so PluginModeProvider receives the live accessToken from auth context,
// which means plugin data refreshes on login/logout without stale cookie reads.
function PluginModeProviderWithAuth({ children }: { children: React.ReactNode }) {
  const { accessToken } = useAuth();
  return <PluginModeProvider accessToken={accessToken}>{children}</PluginModeProvider>;
}

function DashboardShell({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const pathname = usePathname();
  const { accessToken } = useAuth();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const { mode } = usePluginMode();

  const page = legacyKeyForPathname(pathname) || searchParams.get("page") || "api-keys";
  const isGateway = mode === "ai-gateway";

  const navigateToPage = (newPage: string) => {
    const migratedRoute = MIGRATED_PAGES[newPage];
    router.push(migratedRoute ? migratedHref(migratedRoute) : legacyPageHref(newPage));
  };

  // Non-gateway (agent control plane) mode keeps the original full-width Navbar,
  // which carries the account menu; the redesigned sidebar + header shell is
  // scoped to the ai-gateway dashboard. Chat and the public model hub are
  // separate routes that likewise keep the old Navbar.
  if (!isGateway) {
    return (
      <div className="flex h-screen flex-col overflow-hidden bg-background">
        <Navbar accessToken={accessToken} isPublicPage={false} />
        <DebugWarningBanner accessToken={accessToken} />
        <LicenseExpiryBanner accessToken={accessToken} />
        <main className="flex min-h-0 flex-1 overflow-hidden">
          <AgentControlPlaneView />
        </main>
      </div>
    );
  }

  // Standard app shell: the viewport is fixed height and never scrolls. The
  // sidebar owns its own scroll and the content column scrolls independently,
  // so the page can't be dragged past the end of the nav.
  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <SidebarProvider
        setPage={navigateToPage}
        defaultSelectedKey={page}
        sidebarCollapsed={sidebarCollapsed}
        onToggleCollapsed={() => setSidebarCollapsed((v) => !v)}
      />
      <div className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <DashboardHeader page={page} />
        <DebugWarningBanner accessToken={accessToken} />
        <LicenseExpiryBanner accessToken={accessToken} />
        <main className="min-w-0 flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

function LayoutContent({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { accessToken, authLoading } = useAuth();
  const isInvitationFlow = Boolean(searchParams.get("invitation_id"));

  // Legacy invitation links point at /ui/?invitation_id=; the onboarding form now lives at its own
  // /onboarding route. Redirect once ui-config has loaded so migratedHref resolves the SERVER_ROOT_PATH base.
  useEffect(() => {
    if (!authLoading && isInvitationFlow) {
      router.replace(`${migratedHref("onboarding")}?${searchParams.toString()}`);
    }
  }, [authLoading, isInvitationFlow, router, searchParams]);

  if (authLoading || isInvitationFlow) {
    return <LoadingScreen />;
  }

  return (
    <ThemeProvider accessToken={accessToken}>
      <DashboardShell>{children}</DashboardShell>
    </ThemeProvider>
  );
}

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<LoadingScreen />}>
      <PluginModeProviderWithAuth>
        <LayoutContent>{children}</LayoutContent>
      </PluginModeProviderWithAuth>
    </Suspense>
  );
}
