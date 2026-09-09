# 企查查 MCP 与统一计费

将 `config.yaml` 的 `mcp_servers` 合并到网关配置。服务器环境变量 `QCC_API_KEY` 填企查查 Key 的原始值，不包含 `Bearer `；LiteLLM 自动发送 Bearer 认证头。不要把企查查 Key 当成下游 LiteLLM Key。

管理界面：MCP Servers -> Add New MCP Server，搜索“企查查”或 `qcc`，选择服务、填写 Authentication Value 后保存。预设会带入 HTTP transport、Bearer 认证、300 秒超时和工具计费。已有企查查 URL 配置在加载时也会自动获得价格；显式单工具改价优先。

## 价格规则

按照本项目的网关销售规则，1 积分 = 1 USD。这是网关单价，供应商充值比例、赠送积分和按主体月度封顶仍由企查查结算，不在 LiteLLM 中独立计算。

企查查响应顶层 `_extra.settlement` 为 `free_` 开头的非空原因时，该次调用记 0 USD，优先于固定单价、按用量计价和 post-call hook 的费用。例如真实返回的 `free_no_match` 表示未匹配到主体，不扣费；`isError: false` 不代表厂商收费。该规则只用于服务端解析出的企查查 HTTPS 来源，其他 MCP 不受影响。工具正文、`structuredContent` 或请求参数中的同名字段不作为结算依据。未知、缺失或格式错误的标记保持配置价格，不凭“无数据”文字、空列表或缓存命中推断免费。

月度封顶后的免费调用只有在企查查返回上述明确免费标记时才自动免单；不能仅凭账户赠送积分或充值优惠推算实际成本。尚未获得所有结算状态的公开机器协议，非零优惠金额对账仍需供应商提供实际扣分。

价格快照核对于 2026-09-09。企业类 185 个工具、法律类 10 个工具、标讯类 6 个工具按工具单价收费；文档解析每页 1 USD，查询解析结果免费。例如企业工商信息 3 USD/次、股东信息 20 USD/次、企业简介 1 USD/次。

按次价格来自 `litellm/proxy/qcc_mcp_catalog.json`。升级价格快照后重新加载网关；已保存到数据库的显式改价保持优先。每个企查查服务默认要求工具有价格，新增未定价工具会在调用上游前返回 `mcp_tool_price_missing`，避免意外免费调用。

所有 MCP 服务均可配置：

```yaml
mcp_info:
  mcp_server_cost_info:
    require_tool_pricing: true
    default_cost_per_query: 1
    tool_name_to_cost_per_query:
      search: 3
      health: 0
    tool_name_to_cost_per_unit:
      parse_document:
        cost_per_unit: 1
        unit: page
        usage_path: details.*.total_pages
```

错误结果、企查查明确免费结果不计费。其余结果的价格优先级为管理员 post-call hook 的 `response_cost`、单工具按次价格、单工具用量价格、服务默认单价。明确的 0 表示免费；空值不覆盖默认价。负数、布尔值、NaN、无穷和非数值价格被拒绝。

用量路径读取 MCP `structuredContent`，没有该字段时读取唯一的 JSON 文本内容块。点号表示对象字段，`*` 表示对数组各项求和。上例会将 `details` 中各文档的 `total_pages` 相加。缺失或非法用量不会被当成有效的零用量。

## 文档解析

远程 `qcc_document` 接收公开可访问的文档 URL，不能读取用户电脑上的文件。网关提交 `parse_document` 时设置 `wait: true`；若上游仍返回 `processing`，网关使用上游返回的任务 ID 继续查询，直到完成或达到配置的请求超时。完成后以返回的 `details[].total_pages` 计费，并将费用归到原始 `parse_document` 请求。内部轮询和后续 `get_parse_result` 不产生第二笔解析费用。

这意味着通过网关提交文档会等待完成，即使客户端传了 `wait: false`。保持请求连接，并按文档大小设置 `timeout`（示例为 300 秒）。上游失败、无页数、错误的任务 ID 都作为 MCP `isError` 返回，不按成功解析计费。超时或进程中断后的上游任务可能仍在运行；当前没有后台任务对账，供应商成本与网关成功调用账单可能因此不同。

若原始解析调用明确免费，不要求返回页数；若原始解析收费，内部 `get_parse_result` 的免费结算标记不会覆盖原始解析结算，完成后仍按实际页数计费。

## 权限与账本

费用使用现有 LiteLLM MCP spend logging，归属调用的 Virtual Key、用户和 Team；已有预算与权限检查继续生效。使用额度限制需要按原网关部署方式启用数据库和 spend tracking。REST `/mcp-rest/tools/call`、原生 MCP `/mcp` 以及 LiteLLM 执行的 Responses/Chat MCP 调用共用价格规则。MCP 工具费用与 LLM token 费用分别记账。

历史存档服务 `qcc_history` 需要企查查企业实名认证。本次提供的凭证可访问另外 9 个服务，历史存档未通过上游鉴权；完成企查查认证后可以沿用配置。

## 验证

查询服务 ID：`GET /v1/mcp/server`。使用 LiteLLM 的管理 Key 或具有该 MCP 权限的 Virtual Key 调用：

```bash
curl http://localhost:4000/mcp-rest/tools/call \
  -H "Authorization: Bearer $LITELLM_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"server_id":"qcc_company","name":"get_company_profile","arguments":{"searchKey":"企查查科技股份有限公司"}}'
```

`server_id` 需替换为列表返回的实际 ID。在 Logs 中筛选 MCP 调用，检查成功调用的 `response_cost`，上述企业简介应为 1 USD。

官方来源：[接入指南](https://agent.qcc.com/guide)、[工具价格](https://agent.qcc.com/api/tool-config/short-descriptions)、[法律数据](https://agent.qcc.com/legal)、[标讯数据](https://agent.qcc.com/tender)、[文档解析](https://agent.qcc.com/doc)、[积分说明](https://agent.qcc.com/credits)。文档页数结构已用真实远程 MCP 调用验证。
