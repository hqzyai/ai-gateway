import React from "react";
import { Text } from "@tremor/react";
import { MCPServerCostInfo } from "@/components/mcp_tools/types";

interface MCPServerCostDisplayProps {
  costConfig?: MCPServerCostInfo | null;
}

const MCPServerCostDisplay: React.FC<MCPServerCostDisplayProps> = ({ costConfig }) => {
  const hasDefaultCost =
    costConfig?.default_cost_per_query !== undefined && costConfig?.default_cost_per_query !== null;
  const hasToolCosts =
    costConfig?.tool_name_to_cost_per_query && Object.keys(costConfig.tool_name_to_cost_per_query).length > 0;
  const hasCostConfig = hasDefaultCost || hasToolCosts;

  if (!hasCostConfig && !Object.keys(costConfig?.tool_name_to_cost_per_unit || {}).length) {
    return (
      <div className="mt-6 pt-6 border-t border-gray-200">
        <div className="space-y-4">
          <div className="p-4 bg-gray-50 border border-gray-200 rounded-lg">
            <Text className="text-gray-600">
              No cost configuration set for this server. Tool calls will be charged at $0.00 per tool call.
            </Text>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-6 pt-6 border-t border-gray-200">
      <div className="space-y-4">
        {Object.entries(costConfig?.tool_name_to_cost_per_unit || {}).map(([name, pricing]) => (
          <div key={name} className="rounded-lg bg-gray-50 p-3">
            <Text className="font-medium break-all">{name}</Text>
            <Text>
              ${pricing.cost_per_unit.toFixed(4)} per {pricing.unit}
            </Text>
            <Text className="text-gray-500 text-sm">Charged using actual usage in the tool response</Text>
          </div>
        ))}
        {hasDefaultCost &&
          costConfig?.default_cost_per_query !== undefined &&
          costConfig?.default_cost_per_query !== null && (
            <div>
              <Text className="font-medium">Default Cost per Query</Text>
              <div className="text-green-600 font-mono">${costConfig.default_cost_per_query.toFixed(4)}</div>
            </div>
          )}

        {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
          <div>
            <Text className="font-medium">Tool-Specific Costs</Text>
            <div className="mt-2 space-y-2">
              {Object.entries(costConfig.tool_name_to_cost_per_query).map(
                ([toolName, cost]) =>
                  cost !== null &&
                  cost !== undefined && (
                    <div key={toolName} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                      <Text className="font-medium">{toolName}</Text>
                      <Text className="text-green-600 font-mono">${cost.toFixed(4)} per query</Text>
                    </div>
                  ),
              )}
            </div>
          </div>
        )}

        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <Text className="text-blue-800 font-medium">Cost Summary:</Text>
          <div className="mt-2 space-y-1">
            {hasDefaultCost &&
              costConfig?.default_cost_per_query !== undefined &&
              costConfig?.default_cost_per_query !== null && (
                <Text className="text-blue-700">
                  • Default cost: ${costConfig.default_cost_per_query.toFixed(4)} per query
                </Text>
              )}
            {hasToolCosts && costConfig?.tool_name_to_cost_per_query && (
              <Text className="text-blue-700">
                • {Object.keys(costConfig.tool_name_to_cost_per_query).length} tool(s) with custom pricing
              </Text>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MCPServerCostDisplay;
