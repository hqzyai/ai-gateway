"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { all_admin_roles } from "@/utils/roles";

import TokenAnalyticsView from "./_components/TokenAnalyticsView";

export default function TokenAnalyticsPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  const effectiveUserId = all_admin_roles.includes(userRole) ? null : userId;
  return <TokenAnalyticsView accessToken={accessToken} userId={effectiveUserId} />;
}
