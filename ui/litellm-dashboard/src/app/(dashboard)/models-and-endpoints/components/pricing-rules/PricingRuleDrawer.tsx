import { DeleteOutlined, PlusOutlined } from "@ant-design/icons";
import { Alert, Button, Divider, Drawer, Empty, Input, InputNumber, Select, Space, Tabs, Tag, Typography } from "antd";
import { useMemo, useState } from "react";
import { getModeLabel, getPricingDefinition, getVideoPricingScale, pricingDimensions } from "./pricingDimensions";
import { ImagePricingRow, PricingRuleDraft, TieredPricingRow, VideoPricingRow, VideoPricingUnit } from "./types";

const { Text, Title } = Typography;

interface PricingRuleDrawerProps {
  open: boolean;
  initialDraft: PricingRuleDraft | null;
  isOverride: boolean;
  saving: boolean;
  readOnly: boolean;
  onClose: () => void;
  onSave: (draft: PricingRuleDraft) => void;
  onDelete: (() => void) | null;
}

const modes = [
  "chat",
  "responses",
  "embedding",
  "image_generation",
  "video_generation",
  "audio_speech",
  "audio_transcription",
  "rerank",
  "search",
  "ocr",
];

const createId = () => globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;

const PricingRuleDrawer = ({
  open,
  initialDraft,
  isOverride,
  saving,
  readOnly,
  onClose,
  onSave,
  onDelete,
}: PricingRuleDrawerProps) => {
  const [draft, setDraft] = useState<PricingRuleDraft | null>(initialDraft);
  const [customKey, setCustomKey] = useState("");

  const groupedOptions = useMemo(
    () =>
      [...new Set(pricingDimensions.map((dimension) => dimension.category))].map((label) => ({
        label,
        options: pricingDimensions
          .filter((dimension) => dimension.category === label)
          .map((dimension) => ({ label: dimension.label, value: dimension.key })),
      })),
    [],
  );

  if (!draft) return null;

  const updateDraft = (values: Partial<PricingRuleDraft>) =>
    setDraft((current) => (current ? { ...current, ...values } : current));

  const addDimension = (key: string) => {
    if (!key || draft.dimensions.some((row) => row.key === key)) return;
    updateDraft({ dimensions: [...draft.dimensions, { id: createId(), key, value: 0 }] });
  };

  const updateDimension = (id: string, value: number | null) =>
    updateDraft({
      dimensions: draft.dimensions.map((row) => (row.id === id ? { ...row, value: value ?? 0 } : row)),
    });

  const addVideoPricing = () =>
    updateDraft({
      videoPricing: [
        ...draft.videoPricing,
        { id: createId(), inputType: "no_video_input", resolution: "default", value: 0 },
      ],
    });

  const addImagePricing = () =>
    updateDraft({
      imagePricing: [...draft.imagePricing, { id: createId(), direction: "output", resolution: "1k", value: 0 }],
    });

  const updateImagePricing = (id: string, values: Partial<ImagePricingRow>) =>
    updateDraft({ imagePricing: draft.imagePricing.map((row) => (row.id === id ? { ...row, ...values } : row)) });

  const updateVideoPricing = (id: string, values: Partial<VideoPricingRow>) =>
    updateDraft({ videoPricing: draft.videoPricing.map((row) => (row.id === id ? { ...row, ...values } : row)) });

  const updateVideoPricingUnit = (videoPricingUnit: VideoPricingUnit) => {
    const scale = getVideoPricingScale(videoPricingUnit) / getVideoPricingScale(draft.videoPricingUnit);
    updateDraft({
      videoPricingUnit,
      videoPricing: draft.videoPricing.map((row) => ({ ...row, value: row.value * scale })),
    });
  };

  const addTier = () =>
    updateDraft({
      tiers: [...draft.tiers, { id: createId(), lower: 0, upper: null, input: null, output: null, cacheRead: null }],
    });

  const updateTier = (id: string, values: Partial<TieredPricingRow>) =>
    updateDraft({ tiers: draft.tiers.map((tier) => (tier.id === id ? { ...tier, ...values } : tier)) });

  const pricingTab = (
    <div className="space-y-4">
      <Alert
        type="info"
        showIcon
        message={
          draft.mode === "video_generation"
            ? "视频按成功输出的成片时长计费。请添加“视频按秒”维度，并按分辨率填写美元 / 秒。"
            : "Token 价格按每百万 Token 填写，LiteLLM 保存时会自动换算为单个 Token 的价格。"
        }
      />
      <Select
        className="w-full"
        showSearch
        placeholder="添加计费维度"
        options={groupedOptions}
        value={null}
        disabled={readOnly}
        onChange={addDimension}
      />
      <Space.Compact className="w-full">
        <Input
          placeholder="自定义 LiteLLM 字段，例如 output_cost_per_second_4k"
          value={customKey}
          disabled={readOnly}
          onChange={(event) => setCustomKey(event.target.value)}
        />
        <Button
          disabled={readOnly || !customKey.trim()}
          onClick={() => {
            addDimension(customKey.trim());
            setCustomKey("");
          }}
        >
          添加自定义维度
        </Button>
      </Space.Compact>
      {draft.dimensions.length === 0 ? (
        <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="暂未配置固定价格维度" />
      ) : (
        <div className="space-y-2">
          {draft.dimensions.map((row) => {
            const definition = getPricingDefinition(row.key);
            return (
              <div
                key={row.id}
                className="grid grid-cols-[minmax(0,1fr)_180px_36px] items-center gap-3 rounded-lg border border-gray-200 bg-white p-3"
              >
                <div className="min-w-0">
                  <Text strong>{definition.label}</Text>
                  <div className="truncate font-mono text-xs text-gray-500">{row.key}</div>
                </div>
                <InputNumber
                  className="w-full"
                  min={0}
                  precision={9}
                  value={row.value}
                  addonAfter={<span className="whitespace-nowrap text-xs">{definition.unit}</span>}
                  disabled={readOnly}
                  onChange={(value) => updateDimension(row.id, value)}
                />
                <Button
                  type="text"
                  danger
                  aria-label={`删除 ${definition.label}`}
                  icon={<DeleteOutlined />}
                  disabled={readOnly}
                  onClick={() => updateDraft({ dimensions: draft.dimensions.filter((item) => item.id !== row.id) })}
                />
              </div>
            );
          })}
        </div>
      )}
    </div>
  );

  const videoTab = (
    <div className="space-y-4">
      <Alert
        type="info"
        showIcon
        message={
          draft.videoPricingUnit === "per_generation"
            ? "按输入场景和生成规格配置单个生成结果的价格，最终费用为单价乘以生成数量。"
            : "按输入场景和分辨率配置视频 Token 价格，支持 720p、1080p、4k 等自定义分辨率。"
        }
      />
      <div>
        <Text type="secondary">计费单位</Text>
        <Select
          className="mt-1 w-full"
          value={draft.videoPricingUnit}
          disabled={readOnly}
          options={[
            { label: "每百万视频 Token", value: "per_token" },
            { label: "每个生成结果（3D / 资产）", value: "per_generation" },
          ]}
          onChange={updateVideoPricingUnit}
        />
      </div>
      {draft.videoPricing.map((row) => (
        <div
          key={row.id}
          className="grid grid-cols-[170px_120px_minmax(180px,1fr)_36px] items-center gap-2 rounded-lg border border-gray-200 p-3"
        >
          <Select
            value={row.inputType}
            disabled={readOnly}
            options={[
              {
                label: draft.videoPricingUnit === "per_generation" ? "无参考媒体输入" : "不含视频输入",
                value: "no_video_input",
              },
              {
                label: draft.videoPricingUnit === "per_generation" ? "含参考媒体输入" : "包含视频输入",
                value: "video_input",
              },
            ]}
            onChange={(inputType) => updateVideoPricing(row.id, { inputType })}
          />
          <Input
            value={row.resolution}
            placeholder={
              draft.videoPricingUnit === "per_generation"
                ? "standard_no_texture / ultra_hd_texture"
                : "default / 1080p / 4k"
            }
            disabled={readOnly}
            onChange={(event) => updateVideoPricing(row.id, { resolution: event.target.value })}
          />
          <InputNumber
            className="w-full"
            min={0}
            precision={9}
            value={row.value}
            addonAfter={draft.videoPricingUnit === "per_generation" ? "美元 / 个" : "美元 / 百万视频 Token"}
            disabled={readOnly}
            onChange={(value) => updateVideoPricing(row.id, { value: value ?? 0 })}
          />
          <Button
            type="text"
            danger
            aria-label="删除视频价格"
            icon={<DeleteOutlined />}
            disabled={readOnly}
            onClick={() => updateDraft({ videoPricing: draft.videoPricing.filter((item) => item.id !== row.id) })}
          />
        </div>
      ))}
      <Button icon={<PlusOutlined />} disabled={readOnly} onClick={addVideoPricing}>
        添加视频场景
      </Button>
    </div>
  );

  const imageTab = (
    <div className="space-y-4">
      <Alert
        type="info"
        showIcon
        message="按输入 / 输出方向和分辨率档配置每张图片的价格。分辨率档可填写 1k、2k、4k 等供应商返回的标识。"
      />
      {draft.imagePricing.map((row) => (
        <div
          key={row.id}
          className="grid grid-cols-[140px_120px_minmax(180px,1fr)_36px] items-center gap-2 rounded-lg border border-gray-200 p-3"
        >
          <Select
            value={row.direction}
            disabled={readOnly}
            options={[
              { label: "输入图片", value: "input" },
              { label: "输出图片", value: "output" },
            ]}
            onChange={(direction) => updateImagePricing(row.id, { direction })}
          />
          <Input
            value={row.resolution}
            placeholder="1k / 2k / 4k"
            disabled={readOnly}
            onChange={(event) => updateImagePricing(row.id, { resolution: event.target.value })}
          />
          <InputNumber
            className="w-full"
            min={0}
            precision={9}
            value={row.value}
            addonAfter="美元 / 张"
            disabled={readOnly}
            onChange={(value) => updateImagePricing(row.id, { value: value ?? 0 })}
          />
          <Button
            type="text"
            danger
            aria-label="删除图片分辨率价格"
            icon={<DeleteOutlined />}
            disabled={readOnly}
            onClick={() => updateDraft({ imagePricing: draft.imagePricing.filter((item) => item.id !== row.id) })}
          />
        </div>
      ))}
      <Button icon={<PlusOutlined />} disabled={readOnly} onClick={addImagePricing}>
        添加图片分辨率价格
      </Button>
    </div>
  );

  const tierTab = (
    <div className="space-y-4">
      <Alert
        type="info"
        showIcon
        message="系统根据输入长度匹配阶梯计费。区间使用输入 Token 数量，价格使用每百万 Token 的美元价格。"
      />
      {draft.tiers.map((tier) => (
        <div key={tier.id} className="rounded-lg border border-gray-200 p-3">
          <div className="mb-3 grid grid-cols-[1fr_1fr_36px] gap-2">
            <InputNumber
              className="w-full"
              min={0}
              precision={0}
              value={tier.lower}
              addonBefore="从"
              addonAfter="Token"
              disabled={readOnly}
              onChange={(value) => updateTier(tier.id, { lower: value ?? 0 })}
            />
            <InputNumber
              className="w-full"
              min={0}
              precision={0}
              value={tier.upper}
              placeholder="无上限"
              addonBefore="至"
              addonAfter="Token"
              disabled={readOnly}
              onChange={(value) => updateTier(tier.id, { upper: value })}
            />
            <Button
              type="text"
              danger
              aria-label="删除价格阶梯"
              icon={<DeleteOutlined />}
              disabled={readOnly}
              onClick={() => updateDraft({ tiers: draft.tiers.filter((item) => item.id !== tier.id) })}
            />
          </div>
          <div className="grid grid-cols-3 gap-2">
            <InputNumber
              className="w-full"
              min={0}
              precision={9}
              value={tier.input}
              placeholder="输入价格"
              addonAfter="输入 / 百万"
              disabled={readOnly}
              onChange={(value) => updateTier(tier.id, { input: value })}
            />
            <InputNumber
              className="w-full"
              min={0}
              precision={9}
              value={tier.output}
              placeholder="输出价格"
              addonAfter="输出 / 百万"
              disabled={readOnly}
              onChange={(value) => updateTier(tier.id, { output: value })}
            />
            <InputNumber
              className="w-full"
              min={0}
              precision={9}
              value={tier.cacheRead}
              placeholder="缓存读取价格"
              addonAfter="缓存 / 百万"
              disabled={readOnly}
              onChange={(value) => updateTier(tier.id, { cacheRead: value })}
            />
          </div>
        </div>
      ))}
      <Button icon={<PlusOutlined />} disabled={readOnly} onClick={addTier}>
        添加 Token 阶梯
      </Button>
    </div>
  );

  return (
    <Drawer
      open={open}
      width={820}
      onClose={onClose}
      title={
        <Space>
          <span>{isOverride ? "编辑自定义计费规则" : "新增自定义计费规则"}</span>
          {isOverride && <Tag color="purple">自定义</Tag>}
        </Space>
      }
      extra={
        <Space>
          {onDelete && (
            <Button danger disabled={readOnly || saving} onClick={onDelete}>
              恢复上游价格
            </Button>
          )}
          <Button type="primary" loading={saving} disabled={readOnly} onClick={() => onSave(draft)}>
            保存规则
          </Button>
        </Space>
      }
    >
      {readOnly && (
        <Alert className="mb-4" type="warning" showIcon message="只读管理员可以查看计费规则，但不能进行修改。" />
      )}
      <div className="grid grid-cols-2 gap-4">
        <div className="col-span-2">
          <Text type="secondary">模型标识</Text>
          <Input
            className="mt-1"
            value={draft.modelName}
            disabled={readOnly || isOverride}
            onChange={(event) => updateDraft({ modelName: event.target.value })}
          />
        </div>
        <div>
          <Text type="secondary">供应商</Text>
          <Input
            className="mt-1"
            value={draft.provider}
            disabled={readOnly}
            onChange={(event) => updateDraft({ provider: event.target.value })}
          />
        </div>
        <div>
          <Text type="secondary">模式</Text>
          <Select
            className="mt-1 w-full"
            value={draft.mode}
            options={modes.map((mode) => ({ label: getModeLabel(mode), value: mode }))}
            disabled={readOnly}
            onChange={(mode) => updateDraft({ mode })}
          />
        </div>
        <div className="col-span-2">
          <Text type="secondary">价格来源网址</Text>
          <Input
            className="mt-1"
            value={draft.source}
            placeholder="https://provider.example/pricing"
            disabled={readOnly}
            onChange={(event) => updateDraft({ source: event.target.value })}
          />
        </div>
      </div>
      <Divider />
      <Title level={5}>计费维度</Title>
      <Tabs
        items={[
          { key: "flat", label: `固定价格（${draft.dimensions.length}）`, children: pricingTab },
          { key: "image", label: `图片分辨率（${draft.imagePricing.length}）`, children: imageTab },
          {
            key: "video",
            label: `${draft.videoPricingUnit === "per_generation" ? "生成资产" : "视频 Token"}（${draft.videoPricing.length}）`,
            children: videoTab,
          },
          { key: "tiers", label: `Token 阶梯（${draft.tiers.length}）`, children: tierTab },
        ]}
      />
    </Drawer>
  );
};

export default PricingRuleDrawer;
