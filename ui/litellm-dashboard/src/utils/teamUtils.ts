import { Team } from "@/components/key_team_helpers/key_list";

export interface TeamLabel {
  team_id?: string | null;
  team_alias?: string | null;
  metadata?: unknown;
}

export const managedTeamName = (team: TeamLabel | null | undefined): string | undefined => {
  const metadata = team?.metadata;
  if (!metadata || typeof metadata !== "object" || !("display_name" in metadata)) return undefined;
  const name = metadata.display_name;
  return typeof name === "string" && name.trim() ? name : undefined;
};

export const teamDisplayName = (team: TeamLabel | null | undefined): string =>
  managedTeamName(team) || team?.team_alias || team?.team_id || "";

export const matchesTeamSearch = (team: TeamLabel, search: string): boolean =>
  [managedTeamName(team), team.team_alias, team.team_id].some((name) =>
    name?.toLowerCase().includes(search.toLowerCase()),
  );

/**
 * Creates a map from team_id to team_alias for efficient lookups.
 * @param teams - Array of Team objects
 * @returns Record mapping team_id to team_alias
 */
export const createTeamAliasMap = (teams: Team[] | null | undefined): Record<string, string> => {
  if (!teams) return {};
  return teams.reduce(
    (acc, team) => {
      acc[team.team_id] = teamDisplayName(team);
      return acc;
    },
    {} as Record<string, string>,
  );
};

/**
 * Resolves a team alias from a team ID.
 * @param teamID - The team ID to look up
 * @param teams - Array of Team objects
 * @returns The team alias if found, null otherwise
 */
export const resolveTeamAliasFromTeamID = (teamID: string, teams: Team[]): string | null => {
  const team = teams.find((team) => team.team_id === teamID);
  return team ? teamDisplayName(team) : null;
};
