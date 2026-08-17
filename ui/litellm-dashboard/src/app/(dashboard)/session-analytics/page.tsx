"use client";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { all_admin_roles } from "@/utils/roles";

import SessionAnalyticsView from "./_components/SessionAnalyticsView";

export default function SessionAnalyticsPage() {
  const { accessToken, userId, userRole } = useAuthorized();
  const effectiveUserId = all_admin_roles.includes(userRole) ? null : userId;
  return <SessionAnalyticsView accessToken={accessToken} userId={effectiveUserId} />;
}
