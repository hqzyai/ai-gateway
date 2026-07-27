import React, { useState, useEffect } from "react";
import { Button, Popconfirm, Modal, InputNumber, Space, Typography, Tag, Card, Tooltip, Divider } from "antd";
import {
  ReloadOutlined,
  ClockCircleOutlined,
  StopOutlined,
  CloudOutlined,
  DatabaseOutlined,
  InfoCircleOutlined,
  WarningOutlined,
} from "@ant-design/icons";
import {
  reloadModelCostMap,
  scheduleModelCostMapReload,
  cancelModelCostMapReload,
  getModelCostMapReloadStatus,
  getModelCostMapSource,
} from "./networking";
import NotificationsManager from "./molecules/notifications_manager";

const { Text } = Typography;

interface ReloadStatus {
  scheduled: boolean;
  interval_hours: number | null;
  last_run: string | null;
  next_run: string | null;
}

interface CostMapSourceInfo {
  source: "local" | "remote";
  url: string | null;
  is_env_forced: boolean;
  fallback_reason: string | null;
  model_count: number;
}

interface PriceDataReloadProps {
  accessToken: string;
  onReloadSuccess?: () => void;
  buttonText?: string;
  showIcon?: boolean;
  size?: "small" | "middle" | "large";
  type?: "primary" | "default" | "dashed" | "link" | "text";
  className?: string;
  compact?: boolean;
}

const PriceDataReload: React.FC<PriceDataReloadProps> = ({
  accessToken,
  onReloadSuccess,
  buttonText = "刷新价格数据",
  showIcon = true,
  size = "middle",
  type = "primary",
  className = "",
  compact = false,
}) => {
  const [isLoading, setIsLoading] = useState(false);
  const [isScheduling, setIsScheduling] = useState(false);
  const [isCancelling, setIsCancelling] = useState(false);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [hours, setHours] = useState<number>(6);
  const [reloadStatus, setReloadStatus] = useState<ReloadStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [sourceInfo, setSourceInfo] = useState<CostMapSourceInfo | null>(null);
  const [loadingSource, setLoadingSource] = useState(false);

  // Fetch status on component mount and periodically
  useEffect(() => {
    fetchReloadStatus();
    fetchSourceInfo();

    // Refresh status every 30 seconds to keep it up to date
    const interval = setInterval(() => {
      fetchReloadStatus();
      fetchSourceInfo();
    }, 30000);

    return () => clearInterval(interval);
  }, [accessToken]);

  const fetchReloadStatus = async () => {
    if (!accessToken) return;

    setLoadingStatus(true);
    try {
      const status = await getModelCostMapReloadStatus(accessToken);
      setReloadStatus(status);
    } catch (error) {
      console.error("Failed to fetch reload status:", error);
      // Set a default status to prevent UI issues
      setReloadStatus({
        scheduled: false,
        interval_hours: null,
        last_run: null,
        next_run: null,
      });
    } finally {
      setLoadingStatus(false);
    }
  };

  const fetchSourceInfo = async () => {
    if (!accessToken) return;

    setLoadingSource(true);
    try {
      const info = await getModelCostMapSource(accessToken);
      setSourceInfo(info);
    } catch (error) {
      console.error("Failed to fetch cost map source info:", error);
    } finally {
      setLoadingSource(false);
    }
  };

  const handleHardRefresh = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("没有可用的访问令牌");
      return;
    }

    setIsLoading(true);
    try {
      const response = await reloadModelCostMap(accessToken);

      if (response.status === "success") {
        NotificationsManager.success(`价格数据刷新成功，已更新 ${response.models_count || 0} 个模型。`);
        onReloadSuccess?.();
        // Refresh status and source info after successful reload
        await fetchReloadStatus();
        await fetchSourceInfo();
      } else {
        NotificationsManager.fromBackend("价格数据刷新失败");
      }
    } catch (error) {
      console.error("Error reloading price data:", error);
      NotificationsManager.fromBackend("价格数据刷新失败，请重试。");
    } finally {
      setIsLoading(false);
    }
  };
  const handleScheduleReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("没有可用的访问令牌");
      return;
    }

    if (hours <= 0) {
      NotificationsManager.fromBackend("刷新间隔必须大于 0 小时");
      return;
    }

    setIsScheduling(true);
    try {
      const response = await scheduleModelCostMapReload(accessToken, hours);

      if (response.status === "success") {
        NotificationsManager.success(`已设置每 ${hours} 小时自动刷新一次`);
        setShowScheduleModal(false);
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("设置定时刷新失败");
      }
    } catch (error) {
      console.error("Error scheduling reload:", error);
      NotificationsManager.fromBackend("设置定时刷新失败，请重试。");
    } finally {
      setIsScheduling(false);
    }
  };

  const handleCancelReload = async () => {
    if (!accessToken) {
      NotificationsManager.fromBackend("没有可用的访问令牌");
      return;
    }

    setIsCancelling(true);
    try {
      const response = await cancelModelCostMapReload(accessToken);

      if (response.status === "success") {
        NotificationsManager.success("已取消定时刷新");
        await fetchReloadStatus();
      } else {
        NotificationsManager.fromBackend("取消定时刷新失败");
      }
    } catch (error) {
      console.error("Error cancelling reload:", error);
      NotificationsManager.fromBackend("取消定时刷新失败，请重试。");
    } finally {
      setIsCancelling(false);
    }
  };

  const formatDateTime = (dateTimeString: string | null) => {
    if (!dateTimeString) return "从未执行";
    try {
      return new Date(dateTimeString).toLocaleString();
    } catch {
      return dateTimeString;
    }
  };

  const getStatusText = () => {
    if (!reloadStatus?.scheduled) return "未设置";
    if (!reloadStatus.last_run) return "就绪";
    return "运行中";
  };

  const getStatusColor = () => {
    if (!reloadStatus?.scheduled) return "default";
    if (!reloadStatus.last_run) return "processing";
    return "success";
  };

  return (
    <div className={className}>
      {/* Action Buttons */}
      <Space direction="horizontal" size="middle" style={{ marginBottom: compact ? 0 : 16 }}>
        {/* Hard Refresh Button - Always visible */}
        <Popconfirm
          title="立即刷新价格数据"
          description="系统将立即从远程数据源获取最新价格，是否继续？"
          onConfirm={handleHardRefresh}
          okText="继续"
          cancelText="取消"
          okButtonProps={{
            style: {
              backgroundColor: "#6366f1",
              borderColor: "#6366f1",
              color: "white",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
              transition: "all 0.2s ease-in-out",
            },
            onMouseEnter: (e) => {
              e.currentTarget.style.backgroundColor = "#4f46e5";
            },
            onMouseLeave: (e) => {
              e.currentTarget.style.backgroundColor = "#6366f1";
            },
          }}
        >
          <Button
            type={type}
            size={size}
            loading={isLoading}
            icon={showIcon ? <ReloadOutlined /> : undefined}
            style={{
              backgroundColor: "#6366f1",
              borderColor: "#6366f1",
              color: "white",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
              transition: "all 0.2s ease-in-out",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = "#4f46e5";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = "#6366f1";
            }}
          >
            {buttonText}
          </Button>
        </Popconfirm>

        {/* Periodic Reload Controls */}
        {!reloadStatus?.scheduled ? (
          <Button
            type="default"
            size={size}
            icon={<ClockCircleOutlined />}
            onClick={() => setShowScheduleModal(true)}
            style={{
              borderColor: "#d9d9d9",
              color: "#6366f1",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
            }}
          >
            设置定时刷新
          </Button>
        ) : (
          <Button
            type="default"
            size={size}
            danger
            icon={<StopOutlined />}
            loading={isCancelling}
            onClick={handleCancelReload}
            style={{
              borderColor: "#ff4d4f",
              color: "#ff4d4f",
              fontWeight: "500",
              borderRadius: "0.375rem",
              padding: "0.375rem 0.75rem",
              height: "auto",
              fontSize: "0.875rem",
              lineHeight: "1.25rem",
            }}
          >
            取消定时刷新
          </Button>
        )}
      </Space>

      {/* Cost Map Source Info Card */}
      {!compact && sourceInfo && (
        <Card
          size="small"
          style={{
            backgroundColor: sourceInfo.source === "remote" ? "#f0f7ff" : "#fff8f0",
            border: `1px solid ${sourceInfo.source === "remote" ? "#bae0ff" : "#ffd591"}`,
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            {/* Header row */}
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              {sourceInfo.source === "remote" ? (
                <CloudOutlined style={{ color: "#1677ff", fontSize: 16 }} />
              ) : (
                <DatabaseOutlined style={{ color: "#fa8c16", fontSize: 16 }} />
              )}
              <Text strong style={{ fontSize: "13px" }}>
                价格数据源
              </Text>
              <Tag
                color={sourceInfo.source === "remote" ? "blue" : "orange"}
                style={{ marginLeft: "auto", fontWeight: 600, textTransform: "uppercase", fontSize: "11px" }}
              >
                {sourceInfo.source === "remote" ? "远程" : "本地"}
              </Tag>
            </div>

            <Divider style={{ margin: "6px 0" }} />

            {/* Model count */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                已加载模型：
              </Text>
              <Text strong style={{ fontSize: "12px" }}>
                {sourceInfo.model_count.toLocaleString()}
              </Text>
            </div>

            {/* URL (when remote or attempted) */}
            {sourceInfo.url && (
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
                <Text type="secondary" style={{ fontSize: "12px", whiteSpace: "nowrap" }}>
                  {sourceInfo.source === "remote" ? "加载地址：" : "尝试地址："}
                </Text>
                <Tooltip title={sourceInfo.url}>
                  <Text
                    style={{
                      fontSize: "11px",
                      maxWidth: 240,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      display: "block",
                      color: "#1677ff",
                      cursor: "default",
                    }}
                  >
                    {sourceInfo.url}
                  </Text>
                </Tooltip>
              </div>
            )}

            {/* Env forced notice */}
            {sourceInfo.is_env_forced && (
              <div style={{ display: "flex", alignItems: "center", gap: 6, marginTop: 2 }}>
                <InfoCircleOutlined style={{ color: "#fa8c16", fontSize: 12 }} />
                <Text type="secondary" style={{ fontSize: "11px" }}>
                  已通过 <code>LITELLM_LOCAL_MODEL_COST_MAP=True</code> 强制使用本地模式
                </Text>
              </div>
            )}

            {/* Fallback reason */}
            {sourceInfo.fallback_reason && (
              <div
                style={{
                  display: "flex",
                  alignItems: "flex-start",
                  gap: 6,
                  backgroundColor: "#fff7e6",
                  border: "1px solid #ffd591",
                  borderRadius: 4,
                  padding: "4px 8px",
                  marginTop: 2,
                }}
              >
                <WarningOutlined style={{ color: "#fa8c16", fontSize: 12, marginTop: 2 }} />
                <Text style={{ fontSize: "11px", color: "#614700" }}>
                  已回退到本地数据：{sourceInfo.fallback_reason}
                </Text>
              </div>
            )}
          </Space>
        </Card>
      )}

      {/* Reload Schedule Status Card */}
      {!compact && reloadStatus && (
        <Card
          size="small"
          style={{
            backgroundColor: "#f8f9fa",
            border: "1px solid #e9ecef",
            borderRadius: 8,
          }}
        >
          <Space direction="vertical" size="small" style={{ width: "100%" }}>
            {reloadStatus.scheduled ? (
              <div>
                <Tag color="green" icon={<ClockCircleOutlined />}>
                  每 {reloadStatus.interval_hours} 小时自动刷新
                </Tag>
              </div>
            ) : (
              <Text type="secondary">未设置定时刷新</Text>
            )}

            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <Text type="secondary" style={{ fontSize: "12px" }}>
                上次执行：
              </Text>
              <Text style={{ fontSize: "12px" }}>{formatDateTime(reloadStatus.last_run)}</Text>
            </div>

            {reloadStatus.scheduled && (
              <>
                {reloadStatus.next_run && (
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <Text type="secondary" style={{ fontSize: "12px" }}>
                      下次执行：
                    </Text>
                    <Text style={{ fontSize: "12px" }}>{formatDateTime(reloadStatus.next_run)}</Text>
                  </div>
                )}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <Text type="secondary" style={{ fontSize: "12px" }}>
                    状态：
                  </Text>
                  <Tag color={getStatusColor()}>{getStatusText()}</Tag>
                </div>
              </>
            )}
          </Space>
        </Card>
      )}

      {/* Schedule Modal */}
      <Modal
        title="设置定时刷新"
        open={showScheduleModal}
        onOk={handleScheduleReload}
        onCancel={() => setShowScheduleModal(false)}
        confirmLoading={isScheduling}
        okText="设置"
        cancelText="取消"
        okButtonProps={{
          style: {
            backgroundColor: "#6366f1",
            borderColor: "#6366f1",
            color: "white",
          },
        }}
      >
        <div style={{ marginBottom: 16 }}>
          <Text>设置价格数据的自动刷新间隔：</Text>
        </div>
        <div style={{ marginBottom: 16 }}>
          <InputNumber
            min={1}
            max={168} // 1 week max
            value={hours}
            onChange={(value) => setHours(value || 6)}
            addonAfter="小时"
            style={{ width: "100%" }}
          />
        </div>
        <div>
          <Text type="secondary">系统将每 {hours} 小时从远程数据源自动获取一次最新价格。</Text>
        </div>
      </Modal>
    </div>
  );
};

export default PriceDataReload;
