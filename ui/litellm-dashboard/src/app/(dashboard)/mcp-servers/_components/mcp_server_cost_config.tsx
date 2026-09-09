import React, { useState } from "react";
import { Tooltip, InputNumber, Collapse, Badge, Input, Button, Checkbox } from "antd";
import { InfoCircleOutlined, DollarOutlined, ToolOutlined } from "@ant-design/icons";
import { Card, Title, Text } from "@tremor/react";
import { MCPServerCostInfo } from "@/components/mcp_tools/types";

interface MCPServerCostConfigProps {
  value?: MCPServerCostInfo;
  onChange?: (value: MCPServerCostInfo) => void;
  tools?: Array<{ name: string; description?: string | null }>;
  disabled?: boolean;
}

const MCPServerCostConfig: React.FC<MCPServerCostConfigProps> = ({
  value = {},
  onChange,
  tools = [],
  disabled = false,
}) => {
  const [newToolName, setNewToolName] = useState("");
  const configuredTools = Array.from(
    new Map<string, { name: string; description?: string | null }>([
      ...Object.keys(value.tool_name_to_cost_per_query || {}).map((name) => [name, { name }] as const),
      ...tools.map((tool) => [tool.name, tool] as const),
    ]).values(),
  );

  const handleDefaultCostChange = (defaultCost: number | null) => {
    const updated = {
      ...value,
      default_cost_per_query: defaultCost,
    };
    onChange?.(updated);
  };

  const handleToolCostChange = (toolName: string, cost: number | null) => {
    const updated = {
      ...value,
      tool_name_to_cost_per_query: { ...value.tool_name_to_cost_per_query, [toolName]: cost },
    };
    onChange?.(updated);
  };

  const addToolPrice = () => {
    const name = newToolName.trim();
    if (!name || configuredTools.some((tool) => tool.name === name)) return;
    handleToolCostChange(name, value.default_cost_per_query ?? 0);
    setNewToolName("");
  };

  return (
    <Card>
      <div className="space-y-6">
        <div className="flex items-center gap-2 mb-4">
          <DollarOutlined className="text-green-600" />
          <Title>Cost Configuration</Title>
          <Tooltip title="Configure costs for this MCP server's tool calls. Set a default rate and per-tool overrides.">
            <InfoCircleOutlined className="text-gray-400" />
          </Tooltip>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Default Cost per Query ($)
              <Tooltip title="Default cost charged for each tool call to this server.">
                <InfoCircleOutlined className="ml-1 text-gray-400" />
              </Tooltip>
            </label>
            <InputNumber
              aria-label="Default cost per query"
              min={0}
              step={0.0001}
              precision={4}
              placeholder="0.0000"
              value={value.default_cost_per_query}
              onChange={handleDefaultCostChange}
              disabled={disabled}
              style={{ width: "200px" }}
              addonBefore="$"
            />
            <Text className="block mt-1 text-gray-500 text-sm">
              Set a default cost for all tool calls to this server
            </Text>
          </div>

          <Checkbox
            checked={value.require_tool_pricing ?? false}
            disabled={disabled}
            onChange={(event) => onChange?.({ ...value, require_tool_pricing: event.target.checked })}
          >
            Require a tool price or default rate before calling upstream
          </Checkbox>

          <div className="flex flex-wrap gap-2">
            <Input
              aria-label="Tool name for pricing"
              placeholder="Upstream tool name"
              value={newToolName}
              onChange={(event) => setNewToolName(event.target.value)}
              onPressEnter={(event) => {
                event.preventDefault();
                addToolPrice();
              }}
              disabled={disabled}
              className="flex-1 min-w-40"
            />
            <Button
              onClick={addToolPrice}
              disabled={
                disabled || !newToolName.trim() || configuredTools.some((tool) => tool.name === newToolName.trim())
              }
            >
              Add Tool Price
            </Button>
          </div>
          <Text className="text-gray-500 text-sm">
            Prices are in USD. For QCC, 1 credit = $1. A price of 0 is free; clearing a price uses the usage rate or
            default.
          </Text>

          {Object.entries(value.tool_name_to_cost_per_unit || {}).map(([name, pricing]) => (
            <div key={name} className="rounded-lg border p-3 space-y-2">
              <Text className="font-medium break-all">{name} · Usage billing</Text>
              <InputNumber
                aria-label={`Unit cost for ${name}`}
                min={0}
                step={0.0001}
                value={pricing.cost_per_unit}
                addonBefore="$"
                addonAfter={`/${pricing.unit}`}
                disabled={disabled}
                onChange={(cost) => {
                  if (cost === null) return;
                  onChange?.({
                    ...value,
                    tool_name_to_cost_per_unit: {
                      ...value.tool_name_to_cost_per_unit,
                      [name]: { ...pricing, cost_per_unit: cost },
                    },
                  });
                }}
              />
              <Text className="block text-gray-500 text-sm break-all">
                Uses actual {pricing.unit} usage at {pricing.usage_path}. A tool-specific fixed price overrides this
                rate.
              </Text>
            </div>
          ))}

          {configuredTools.length > 0 && (
            <div className="space-y-4">
              <label className="block text-sm font-medium text-gray-700">
                Tool-Specific Costs ($)
                <Tooltip title="Override the default cost for specific tools. Leave blank to use the default rate.">
                  <InfoCircleOutlined className="ml-1 text-gray-400" />
                </Tooltip>
              </label>
              <Collapse
                items={[
                  {
                    key: "1",
                    label: (
                      <div className="flex items-center">
                        <ToolOutlined className="mr-2 text-blue-500" />
                        <span className="font-medium">Available Tools</span>
                        <Badge
                          count={configuredTools.length}
                          style={{
                            backgroundColor: "#52c41a",
                            marginLeft: "8px",
                          }}
                        />
                      </div>
                    ),
                    children: (
                      <div className="space-y-3 max-h-64 overflow-y-auto">
                        {configuredTools.map((tool) => (
                          <div
                            key={tool.name}
                            className="flex flex-wrap gap-3 items-center justify-between p-3 bg-gray-50 rounded-lg"
                          >
                            <div className="flex-1 min-w-0 break-words">
                              <Text className="font-medium text-gray-900">{tool.name}</Text>
                              {tool.description && (
                                <Text className="text-gray-500 text-sm block mt-1">{tool.description}</Text>
                              )}
                            </div>
                            <div className="ml-4">
                              <InputNumber
                                aria-label={`Cost for ${tool.name}`}
                                min={0}
                                step={0.0001}
                                precision={4}
                                placeholder="Use default"
                                value={value.tool_name_to_cost_per_query?.[tool.name]}
                                onChange={(cost) => handleToolCostChange(tool.name, cost)}
                                disabled={disabled}
                                style={{ width: "120px" }}
                                addonBefore="$"
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    ),
                  },
                ]}
              />
            </div>
          )}
        </div>

        {(value.default_cost_per_query != null ||
          (value.tool_name_to_cost_per_query && Object.keys(value.tool_name_to_cost_per_query).length > 0)) && (
          <div className="mt-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <Text className="text-blue-800 font-medium">Cost Summary:</Text>
            <div className="mt-2 space-y-1">
              {value.default_cost_per_query != null && (
                <Text className="text-blue-700">
                  • Default cost: ${value.default_cost_per_query.toFixed(4)} per query
                </Text>
              )}
              {value.tool_name_to_cost_per_query &&
                Object.entries(value.tool_name_to_cost_per_query).map(
                  ([toolName, cost]) =>
                    cost !== null &&
                    cost !== undefined && (
                      <Text key={toolName} className="text-blue-700">
                        • {toolName}: ${cost.toFixed(4)} per query
                      </Text>
                    ),
                )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPServerCostConfig;
