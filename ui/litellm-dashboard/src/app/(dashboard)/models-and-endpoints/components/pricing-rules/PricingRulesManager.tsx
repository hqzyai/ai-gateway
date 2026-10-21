import PriceDataReload from "@/components/price_data_reload";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { isProxyAdminRole } from "@/utils/roles";
import { isDefaultPricingProvider, isModelProvider } from "@/utils/providerScope";
import { DeleteOutlined, EditOutlined, PlusOutlined, SearchOutlined, SlidersOutlined } from "@ant-design/icons";
import {
  Button,
  Card,
  ConfigProvider,
  Input,
  Popconfirm,
  Select,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
} from "antd";
import type { ColumnsType } from "antd/es/table";
import zhCN from "antd/locale/zh_CN";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { deleteModelCostMapOverride, getModelCostMapOverrides, upsertModelCostMapOverride } from "./api";
import PricingRuleDrawer from "./PricingRuleDrawer";
import {
  createPricingRuleDraft,
  getModeLabel,
  getPricingDefinition,
  getVideoPricingUnit,
  pricingKeysForEntry,
  serializePricingRule,
} from "./pricingDimensions";
import { ModelCostEntry, ModelCostOverride, PricingRuleDraft, VideoPricingUnit } from "./types";

const { Text, Title } = Typography;
const emptyOverrides: readonly ModelCostOverride[] = Object.freeze([]);

const displayDimension = (value: string, videoPricingUnit: VideoPricingUnit) => {
  if (value === "image_resolution_pricing") return "图片分辨率";
  if (value === "video_token_pricing") return videoPricingUnit === "per_generation" ? "生成资产" : "视频 Token";
  if (value === "tiered_pricing") return "Token 阶梯";
  return getPricingDefinition(value).label;
};

interface PricingRulesManagerProps {
  accessToken: string;
  userRole: string;
  modelCostMap: Record<string, ModelCostEntry>;
  loading: boolean;
  onModelCostMapReload: () => void;
}

interface PricingTableRow {
  key: string;
  modelName: string;
  provider: string;
  mode: string;
  dimensions: string[];
  videoPricingUnit: VideoPricingUnit;
  isOverride: boolean;
  entry: ModelCostEntry;
}

const PricingRulesManager = ({
  accessToken,
  userRole,
  modelCostMap,
  loading,
  onModelCostMapReload,
}: PricingRulesManagerProps) => {
  const [search, setSearch] = useState("");
  const [mode, setMode] = useState<string | null>(null);
  const [provider, setProvider] = useState<string | null>(null);
  const [sourceFilter, setSourceFilter] = useState<"all" | "custom">("all");
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [draft, setDraft] = useState<PricingRuleDraft | null>(null);
  const [saving, setSaving] = useState(false);
  const canWrite = isProxyAdminRole(userRole);
  const {
    data: overrideResponse,
    isLoading: loadingOverrides,
    refetch: refetchOverrides,
  } = useQuery({
    queryKey: ["model-cost-map-overrides"],
    queryFn: () => getModelCostMapOverrides(accessToken),
  });
  const overrides = overrideResponse?.overrides ?? emptyOverrides;

  const overridesByModel = useMemo(
    () => new Map(overrides.map((override) => [override.model_name, override.values])),
    [overrides],
  );

  const rows = useMemo<PricingTableRow[]>(
    () =>
      [...new Set([...Object.keys(modelCostMap), ...overridesByModel.keys()])].flatMap((modelName) => {
        const base = modelCostMap[modelName] ?? {};
        const override = overridesByModel.get(modelName);
        const combined = { ...base, ...override };
        const provider =
          typeof combined.litellm_provider === "string" ? combined.litellm_provider : modelName.split("/")[0];
        const isAllowed = isDefaultPricingProvider(provider) || (provider === "openai" && override !== undefined);
        if (modelName === "sample_spec" || modelName === combined.id || !isAllowed) {
          return [];
        }
        const effectiveEntry: ModelCostEntry =
          provider === "openai" ? { ...override, litellm_provider: provider, mode: combined.mode ?? "chat" } : combined;
        const dimensions = pricingKeysForEntry(effectiveEntry);
        if (dimensions.length === 0) return [];
        return [
          {
            key: modelName,
            modelName,
            provider,
            mode: typeof effectiveEntry.mode === "string" ? effectiveEntry.mode : "unknown",
            dimensions,
            videoPricingUnit: getVideoPricingUnit(effectiveEntry),
            isOverride: override !== undefined,
            entry: effectiveEntry,
          },
        ];
      }),
    [modelCostMap, overridesByModel],
  );

  const providers = useMemo(
    () => [...new Set(rows.map((row) => row.provider))].sort().map((value) => ({ value, label: value })),
    [rows],
  );

  const modes = useMemo(
    () => [...new Set(rows.map((row) => row.mode))].sort().map((value) => ({ value, label: getModeLabel(value) })),
    [rows],
  );

  const filteredRows = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    const matchesRow = (row: PricingTableRow) => {
      const matchesSearch =
        !normalizedSearch ||
        row.modelName.toLowerCase().includes(normalizedSearch) ||
        row.provider.toLowerCase().includes(normalizedSearch);
      const matchesMode = !mode || row.mode === mode;
      const matchesProvider = !provider || row.provider === provider;
      const matchesSource = sourceFilter === "all" || row.isOverride;
      return [matchesSearch, matchesMode, matchesProvider, matchesSource].every(Boolean);
    };
    return rows.filter(matchesRow);
  }, [mode, provider, rows, search, sourceFilter]);

  const openRule = (modelName: string) => {
    const row = rows.find((row) => row.modelName === modelName);
    if (!row) return;
    setSelectedModel(modelName);
    setDraft(createPricingRuleDraft(modelName, row.entry));
  };

  const createRule = () => {
    setSelectedModel(null);
    setDraft(createPricingRuleDraft("", { litellm_provider: "siliconflow" }));
  };

  const saveRule = async (nextDraft: PricingRuleDraft) => {
    if (!nextDraft.modelName.trim()) {
      NotificationsManager.fromBackend("请输入模型标识");
      return;
    }
    if (!isModelProvider(nextDraft.provider.trim())) {
      NotificationsManager.fromBackend("仅支持 siliconflow、volcengine、dashscope 和 openai 供应商");
      return;
    }
    if (
      nextDraft.dimensions.length +
        nextDraft.imagePricing.length +
        nextDraft.videoPricing.length +
        nextDraft.tiers.length ===
      0
    ) {
      NotificationsManager.fromBackend("请至少添加一个计费维度");
      return;
    }
    setSaving(true);
    try {
      await upsertModelCostMapOverride(accessToken, {
        model_name: nextDraft.modelName.trim(),
        values: serializePricingRule(nextDraft),
      });
      NotificationsManager.success(`已保存 ${nextDraft.modelName} 的计费规则`);
      setDraft(null);
      await refetchOverrides();
      onModelCostMapReload();
    } catch (error) {
      NotificationsManager.fromBackend(error);
    } finally {
      setSaving(false);
    }
  };

  const deleteRule = async (modelName: string) => {
    setSaving(true);
    try {
      await deleteModelCostMapOverride(accessToken, modelName);
      NotificationsManager.success(`已删除 ${modelName} 的自定义计费规则`);
      setDraft(null);
      await refetchOverrides();
      onModelCostMapReload();
    } catch (error) {
      NotificationsManager.fromBackend(error);
    } finally {
      setSaving(false);
    }
  };

  const columns: ColumnsType<PricingTableRow> = [
    {
      title: "模型",
      dataIndex: "modelName",
      width: "36%",
      render: (value: string, row) => (
        <div>
          <Space size={6}>
            <Text strong>{value}</Text>
            {row.isOverride && <Tag color="purple">自定义</Tag>}
          </Space>
          <div className="mt-1 text-xs text-gray-500">{row.provider}</div>
        </div>
      ),
    },
    {
      title: "模式",
      dataIndex: "mode",
      width: 150,
      render: (value: string) => <Tag>{getModeLabel(value)}</Tag>,
    },
    {
      title: "计费维度",
      dataIndex: "dimensions",
      render: (values: string[], row) => (
        <Space size={[4, 4]} wrap>
          {values.slice(0, 3).map((value) => (
            <Tag key={value} color="blue">
              {displayDimension(value, row.videoPricingUnit)}
            </Tag>
          ))}
          {values.length > 3 && <Tag>+{values.length - 3}</Tag>}
          {values.length === 0 && <Text type="secondary">暂无价格</Text>}
        </Space>
      ),
    },
    {
      title: "操作",
      width: 130,
      align: "right",
      render: (_, row) => (
        <Space>
          <Button type="text" icon={<EditOutlined />} onClick={() => openRule(row.modelName)}>
            {canWrite ? "配置" : "查看"}
          </Button>
          {row.isOverride && canWrite && (
            <Popconfirm
              title={row.provider === "openai" ? "删除自定义计费规则？" : "恢复上游价格？"}
              description="这会删除该模型的自定义计费规则。"
              okText={row.provider === "openai" ? "删除" : "恢复"}
              cancelText="取消"
              onConfirm={() => deleteRule(row.modelName)}
            >
              <Button
                type="text"
                danger
                aria-label={`${row.provider === "openai" ? "删除" : "恢复"} ${row.modelName}`}
                icon={<DeleteOutlined />}
              />
            </Popconfirm>
          )}
        </Space>
      ),
    },
  ];

  return (
    <ConfigProvider locale={zhCN}>
      <div className="space-y-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <Title level={3} className="!mb-1">
              模型计费规则
            </Title>
            <Text type="secondary">维护多维度模型价格。刷新上游价格后，系统会自动重新应用自定义规则。</Text>
          </div>
          <Space wrap>
            <PriceDataReload
              accessToken={accessToken}
              onReloadSuccess={onModelCostMapReload}
              buttonText="刷新最新价格"
              size="middle"
              type="default"
              compact
            />
            <Button type="primary" icon={<PlusOutlined />} disabled={!canWrite} onClick={createRule}>
              新增计费规则
            </Button>
          </Space>
        </div>

        <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
          <Card size="small">
            <Statistic
              title="已有价格的模型"
              value={rows.filter((row) => row.dimensions.length > 0).length}
              prefix={<SlidersOutlined />}
            />
          </Card>
          <Card size="small">
            <Statistic
              title="自定义规则"
              value={rows.filter((row) => row.isOverride).length}
              valueStyle={{ color: "#7c3aed" }}
            />
          </Card>
          <Card size="small">
            <Statistic title="计费模式" value={new Set(rows.map((row) => row.mode)).size} />
          </Card>
        </div>

        <Card styles={{ body: { padding: 0 } }}>
          <div className="flex flex-wrap gap-3 border-b border-gray-200 p-4">
            <Input
              allowClear
              className="min-w-64 flex-1"
              prefix={<SearchOutlined />}
              placeholder="搜索模型或供应商"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <Select
              allowClear
              className="w-48"
              placeholder="全部供应商"
              options={providers}
              value={provider}
              onChange={setProvider}
            />
            <Select
              allowClear
              className="w-48"
              placeholder="全部模式"
              options={modes}
              value={mode}
              onChange={setMode}
            />
            <Select
              className="w-44"
              value={sourceFilter}
              options={[
                { value: "all", label: "全部规则来源" },
                { value: "custom", label: "仅自定义规则" },
              ]}
              onChange={setSourceFilter}
            />
          </div>
          <Table
            rowKey="key"
            columns={columns}
            dataSource={filteredRows}
            loading={loading || loadingOverrides}
            pagination={{ pageSize: 12, showSizeChanger: true, showTotal: (total) => `共 ${total} 个模型` }}
            scroll={{ x: 980 }}
          />
        </Card>

        <PricingRuleDrawer
          key={draft?.modelName ?? "closed"}
          open={draft !== null}
          initialDraft={draft}
          isOverride={selectedModel !== null && overridesByModel.has(selectedModel)}
          saving={saving}
          readOnly={!canWrite}
          onClose={() => setDraft(null)}
          onSave={saveRule}
          onDelete={
            selectedModel !== null && overridesByModel.has(selectedModel) ? () => deleteRule(selectedModel) : null
          }
        />
      </div>
    </ConfigProvider>
  );
};

export default PricingRulesManager;
