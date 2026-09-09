import React, { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import MCPServerCostConfig from "./mcp_server_cost_config";
import MCPServerCostDisplay from "./mcp_server_cost_display";
import { MCPServerCostInfo } from "@/components/mcp_tools/types";

function ControlledCostConfig({ initial }: { initial: MCPServerCostInfo }) {
  const [value, setValue] = useState(initial);
  return (
    <>
      <MCPServerCostConfig value={value} onChange={setValue} />
      <output data-testid="cost-value">{JSON.stringify(value)}</output>
    </>
  );
}

describe("MCP billing configuration", () => {
  it("edits stored tool prices without a live tool connection", () => {
    render(
      <ControlledCostConfig initial={{ default_cost_per_query: 2, tool_name_to_cost_per_query: { lookup: 3 } }} />,
    );
    fireEvent.click(screen.getByText("Available Tools"));
    expect(screen.getByLabelText("Cost for lookup")).toHaveValue("3.0000");
    fireEvent.change(screen.getByLabelText("Cost for lookup"), { target: { value: "" } });
    expect(JSON.parse(screen.getByTestId("cost-value").textContent || "{}")).toEqual({
      default_cost_per_query: 2,
      tool_name_to_cost_per_query: { lookup: null },
    });
  });

  it("adds a named free tool and requires explicit prices", () => {
    render(<ControlledCostConfig initial={{}} />);
    fireEvent.change(screen.getByLabelText("Tool name for pricing"), { target: { value: "health" } });
    fireEvent.click(screen.getByRole("button", { name: "Add Tool Price" }));
    fireEvent.click(screen.getByRole("checkbox"));
    expect(JSON.parse(screen.getByTestId("cost-value").textContent || "{}")).toEqual({
      tool_name_to_cost_per_query: { health: 0 },
      require_tool_pricing: true,
    });
  });

  it("shows an explicitly free default in the summary", () => {
    render(<MCPServerCostConfig value={{ default_cost_per_query: 0 }} />);
    expect(screen.getByText(/Default cost: \$0.0000 per query/)).toBeInTheDocument();
  });

  it("updates the page rate without losing the usage path or free polling", () => {
    render(
      <ControlledCostConfig
        initial={{
          tool_name_to_cost_per_query: { get_parse_result: 0 },
          tool_name_to_cost_per_unit: {
            parse_document: { unit: "page", cost_per_unit: 1, usage_path: "details.*.total_pages" },
          },
        }}
      />,
    );
    fireEvent.change(screen.getByLabelText("Unit cost for parse_document"), { target: { value: "2" } });
    const value = JSON.parse(screen.getByTestId("cost-value").textContent || "{}");
    expect(value.tool_name_to_cost_per_unit.parse_document).toEqual({
      unit: "page",
      cost_per_unit: 2,
      usage_path: "details.*.total_pages",
    });
    expect(value.tool_name_to_cost_per_query.get_parse_result).toBe(0);
  });

  it("respects disabled pricing controls", () => {
    const onChange = vi.fn();
    render(<MCPServerCostConfig value={{}} disabled onChange={onChange} />);
    expect(screen.getByLabelText("Default cost per query")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add Tool Price" })).toBeDisabled();
    expect(screen.getByRole("checkbox")).toBeDisabled();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("displays a metered-only configuration as paid usage", () => {
    render(
      <MCPServerCostDisplay
        costConfig={{ tool_name_to_cost_per_unit: { parse: { unit: "page", cost_per_unit: 1, usage_path: "pages" } } }}
      />,
    );
    expect(screen.getByText("$1.0000 per page")).toBeInTheDocument();
    expect(screen.queryByText(/No cost configuration/)).not.toBeInTheDocument();
  });
});
