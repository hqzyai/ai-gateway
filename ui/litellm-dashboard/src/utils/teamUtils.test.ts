import { describe, expect, it } from "vitest";
import {
  resolveTeamAliasFromTeamID,
  createTeamAliasMap,
  managedTeamName,
  teamDisplayName,
  matchesTeamSearch,
} from "./teamUtils";
import type { Team } from "@/components/key_team_helpers/key_list";

describe("department display names", () => {
  it("uses the managed label without changing the alias or ID", () => {
    const team = Object.freeze({ team_id: "team-1", team_alias: "engineering", metadata: { display_name: "研发部" } });
    expect(teamDisplayName(team)).toBe("研发部");
    expect(matchesTeamSearch(team, "研发")).toBe(true);
    expect(matchesTeamSearch(team, "ENGINEERING")).toBe(true);
    expect(matchesTeamSearch(team, "team-1")).toBe(true);
    expect(matchesTeamSearch(team, "unknown")).toBe(false);
    expect(team.team_alias).toBe("engineering");
  });

  it.each([undefined, null, {}, { display_name: 10 }, { display_name: " " }])(
    "falls back for malformed metadata %j",
    (metadata) => {
      expect(managedTeamName({ metadata })).toBeUndefined();
      expect(teamDisplayName({ metadata, team_alias: "original", team_id: "id" })).toBe("original");
      expect(teamDisplayName({ metadata, team_id: "id" })).toBe("id");
    },
  );

  it("uses the same names in usage and key lookup maps", () => {
    const teams: Team[] = [
      {
        team_id: "team-1",
        team_alias: "engineering",
        metadata: { display_name: "研发部" },
        models: [],
        max_budget: null,
        budget_duration: null,
        tpm_limit: null,
        rpm_limit: null,
        organization_id: "org",
        created_at: "",
        keys: [],
        members_with_roles: [],
        spend: 0,
      },
    ];
    expect(createTeamAliasMap(teams)).toEqual({ "team-1": "研发部" });
    expect(resolveTeamAliasFromTeamID("team-1", teams)).toBe("研发部");
  });
});

describe("resolveTeamAliasFromTeamID", () => {
  it("should return team alias when team is found", () => {
    const teams = [
      {
        team_id: "team1",
        team_alias: "Team One",
      },
      {
        team_id: "team2",
        team_alias: "Team Two",
      },
    ] as unknown as Team[];

    const result = resolveTeamAliasFromTeamID("team1", teams);
    expect(result).toBe("Team One");
  });

  it("should return null when team is not found", () => {
    const teams = [
      {
        team_id: "team1",
        team_alias: "Team One",
      },
      {
        team_id: "team2",
        team_alias: "Team Two",
      },
    ] as unknown as Team[];

    const result = resolveTeamAliasFromTeamID("team3", teams);
    expect(result).toBeNull();
  });
});

describe("createTeamAliasMap", () => {
  it("should create a map from team_id to team_alias", () => {
    const teams = [
      {
        team_id: "team1",
        team_alias: "Team One",
      },
      {
        team_id: "team2",
        team_alias: "Team Two",
      },
    ] as unknown as Team[];

    const result = createTeamAliasMap(teams);
    expect(result).toEqual({
      team1: "Team One",
      team2: "Team Two",
    });
  });

  it("should return empty object when teams is null or undefined", () => {
    expect(createTeamAliasMap(null)).toEqual({});
    expect(createTeamAliasMap(undefined)).toEqual({});
    expect(createTeamAliasMap([])).toEqual({});
  });
});
