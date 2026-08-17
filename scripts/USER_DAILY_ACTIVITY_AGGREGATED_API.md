# 用户每日用量聚合接口

本文档说明如何调用 LiteLLM 的用户每日用量聚合接口，以及响应中每个字段的含义。该接口适合首页仪表盘、成本优化卡片和用量趋势页；它返回按天汇总的数据，不返回逐请求或逐会话日志。

## 接口概览

```http
GET /user/daily/activity/aggregated
```

接口从每日用户用量聚合表读取数据，在数据库中一次性完成日期、模型、模型组、供应商、API Key、MCP Server 和 Endpoint 等维度的汇总。

主要用途：

- 查询一个日期范围内的总消费、Token 和请求数
- 展示每天的消费与 Token 趋势
- 查看 Prompt 压缩的毛节省、额外开销、净节省和净节省率
- 查看 Prompt 缓存读写 Token 及缓存节省金额
- 按用户、模型或 API Key 筛选
- 按模型、模型组、供应商、Endpoint、MCP Server 和 API Key 查看明细
- 在每个维度下继续查看 API Key 明细

## 鉴权与权限

请求必须带 LiteLLM 访问令牌：

```http
Authorization: Bearer <LITELLM_TOKEN>
```

权限规则：

- 管理员不传 `user_id` 时可以查看全局用户汇总
- 管理员传 `user_id` 时可以查看指定用户
- 非管理员只能查看自己的数据
- 非管理员传入其他用户的 `user_id` 时返回 `403`

不要在文档、日志、前端代码或工单中写入真实令牌。

## 查询参数

| 参数 | 类型 | 必填 | 含义 |
| --- | --- | --- | --- |
| `start_date` | string | 是 | 查询开始日期，格式为 `YYYY-MM-DD` |
| `end_date` | string | 是 | 查询结束日期，格式为 `YYYY-MM-DD`，包含当天 |
| `model` | string | 否 | 只统计指定模型，必须与聚合数据中保存的模型名完全一致 |
| `api_key` | string | 否 | 只统计指定 API Key 标识。数据库通常保存 Key 哈希或内部标识，不要默认传明文 Virtual Key |
| `user_id` | string | 否 | 只统计指定用户；管理员可传任意用户，非管理员只能传自己的用户 ID |
| `timezone` | integer | 否 | UTC 时区偏移分钟数，遵循 JavaScript `Date.getTimezoneOffset()` 约定 |

常见时区值：

| 时区 | `timezone` |
| --- | ---: |
| 中国标准时间 UTC+8 | `-480` |
| UTC | `0` |
| 美国太平洋标准时间 UTC-8 | `480` |

`start_date` 和 `end_date` 缺失时返回 `400`。

## 调用示例

查询上海时区的单日全局汇总：

```bash
curl --get 'http://127.0.0.1:4000/user/daily/activity/aggregated' \
  --header "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  --data-urlencode 'start_date=2026-08-13' \
  --data-urlencode 'end_date=2026-08-13' \
  --data-urlencode 'timezone=-480'
```

查询指定用户：

```bash
curl --get 'http://127.0.0.1:4000/user/daily/activity/aggregated' \
  --header "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  --data-urlencode 'start_date=2026-08-01' \
  --data-urlencode 'end_date=2026-08-31' \
  --data-urlencode 'user_id=user-123' \
  --data-urlencode 'timezone=-480'
```

查询指定模型和 API Key：

```bash
curl --get 'http://127.0.0.1:4000/user/daily/activity/aggregated' \
  --header "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  --data-urlencode 'start_date=2026-08-01' \
  --data-urlencode 'end_date=2026-08-31' \
  --data-urlencode 'model=qwen3.6-27b-nvfp4' \
  --data-urlencode 'api_key=<KEY_HASH_OR_INTERNAL_ID>' \
  --data-urlencode 'timezone=-480'
```

## 响应结构

```json
{
  "results": [
    {
      "date": "2026-08-13",
      "metrics": {
        "spend": 0.0,
        "prompt_tokens": 2223099,
        "completion_tokens": 62115,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "compression_saved_tokens": 237283,
        "compression_gross_saved_tokens": 509365,
        "compression_extra_input_tokens": 272082,
        "compression_savings_spend": 0.0,
        "compression_gross_savings_spend": 0.0,
        "compression_extra_input_spend": 0.0,
        "prompt_caching_savings_spend": 0.0,
        "total_tokens": 2285214,
        "successful_requests": 76,
        "failed_requests": 5,
        "api_requests": 81
      },
      "breakdown": {
        "mcp_servers": {},
        "models": {},
        "model_groups": {},
        "providers": {},
        "endpoints": {},
        "api_keys": {},
        "entities": {}
      }
    }
  ],
  "metadata": {
    "total_spend": 0.0,
    "total_prompt_tokens": 2223099,
    "total_completion_tokens": 62115,
    "total_tokens": 2285214,
    "total_api_requests": 81,
    "total_successful_requests": 76,
    "total_failed_requests": 5,
    "total_cache_read_input_tokens": 0,
    "total_cache_creation_input_tokens": 0,
    "total_compression_saved_tokens": 237283,
    "total_compression_gross_saved_tokens": 509365,
    "total_compression_extra_input_tokens": 272082,
    "total_compression_net_savings_rate": 0.09644152818546063,
    "total_compression_savings_spend": 0.0,
    "total_compression_gross_savings_spend": 0.0,
    "total_compression_extra_input_spend": 0.0,
    "total_prompt_caching_savings_spend": 0.0,
    "page": 1,
    "total_pages": 1,
    "has_more": false
  }
}
```

示例数值仅用于说明结构。实际数据以接口返回为准。

## 顶层字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `results` | array | 按日期排列的每日汇总及维度明细 |
| `metadata` | object | 整个查询日期范围内的总计 |

## `metadata` 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `total_spend` | number | 查询范围内 LiteLLM 记录的总消费金额 |
| `total_prompt_tokens` | integer | 实际计费记录中的 Prompt Token 总量 |
| `total_completion_tokens` | integer | Completion Token 总量 |
| `total_tokens` | integer | `total_prompt_tokens + total_completion_tokens` |
| `total_api_requests` | integer | API 请求总数，包括成功和失败请求 |
| `total_successful_requests` | integer | 成功请求数 |
| `total_failed_requests` | integer | 失败请求数 |
| `total_cache_read_input_tokens` | integer | 从上游模型 Prompt Cache 读取的输入 Token 总量 |
| `total_cache_creation_input_tokens` | integer | 写入或创建上游 Prompt Cache 的输入 Token 总量 |
| `total_compression_saved_tokens` | integer | 压缩净节省 Token，等于毛节省减去额外输入；允许为负数 |
| `total_compression_gross_saved_tokens` | integer | 首轮压缩本身移除的 Prompt Token，不扣除后续召回开销 |
| `total_compression_extra_input_tokens` | integer | 压缩后新增的输入开销，例如 CCR 召回后的二次模型请求 |
| `total_compression_net_savings_rate` | number | 压缩净节省率，范围不强制限制为 0 到 1；负收益时可以小于 0 |
| `total_compression_savings_spend` | number | 压缩净节省金额；负收益时可以为负数 |
| `total_compression_gross_savings_spend` | number | 压缩毛节省 Token 按对应模型输入单价折算的金额 |
| `total_compression_extra_input_spend` | number | 额外输入 Token 按对应模型输入单价折算的金额 |
| `total_prompt_caching_savings_spend` | number | 缓存读取相对普通输入价格节省的金额 |
| `page` | integer | 固定为 `1`，用于兼容分页接口的数据结构 |
| `total_pages` | integer | 固定为 `1` |
| `has_more` | boolean | 固定为 `false`，该接口一次返回完整日期范围 |

## `results[]` 字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `date` | string | 当前汇总日期，格式为 `YYYY-MM-DD` |
| `metrics` | object | 当天全部数据的指标总计 |
| `breakdown` | object | 当天按不同维度拆分后的指标 |

### `metrics` 字段

`results[].metrics`、各个 `breakdown` 实体的 `metrics` 和 `api_key_breakdown` 中的 `metrics` 使用同一套字段。

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `spend` | number | 当前范围或维度的消费金额 |
| `prompt_tokens` | integer | Prompt Token |
| `completion_tokens` | integer | Completion Token |
| `cache_read_input_tokens` | integer | 缓存读取 Token |
| `cache_creation_input_tokens` | integer | 缓存创建 Token |
| `compression_saved_tokens` | integer | 压缩净节省 Token，允许为负数 |
| `compression_gross_saved_tokens` | integer | 压缩毛节省 Token |
| `compression_extra_input_tokens` | integer | 召回及二次请求等额外输入 Token |
| `compression_savings_spend` | number | 压缩净节省金额，允许为负数 |
| `compression_gross_savings_spend` | number | 压缩毛节省金额 |
| `compression_extra_input_spend` | number | 压缩额外输入金额 |
| `prompt_caching_savings_spend` | number | Prompt Cache 节省金额 |
| `total_tokens` | integer | `prompt_tokens + completion_tokens` |
| `successful_requests` | integer | 成功请求数 |
| `failed_requests` | integer | 失败请求数 |
| `api_requests` | integer | 请求总数 |

## `breakdown` 字段

| 字段 | Key 的含义 | 用途 |
| --- | --- | --- |
| `mcp_servers` | MCP namespaced tool name | 查看各 MCP Server 或工具的用量与成本 |
| `models` | 模型名 | 查看各模型的用量、成本和优化收益 |
| `model_groups` | 模型组名 | 查看路由模型组的聚合数据 |
| `providers` | `custom_llm_provider` | 查看 OpenAI、Anthropic 等供应商维度的数据 |
| `endpoints` | HTTP Endpoint | 查看 `/chat/completions`、`/responses` 等接口维度的数据 |
| `api_keys` | API Key 哈希或内部标识 | 查看每个 Key 的用量和成本 |
| `entities` | 实体 ID | 通用实体维度；本聚合接口跨用户折叠数据时通常为空 |

除 `api_keys` 外，每个维度实体通常使用以下结构：

```json
{
  "metrics": {
    "prompt_tokens": 100000,
    "compression_saved_tokens": 12000
  },
  "metadata": {},
  "api_key_breakdown": {
    "<KEY_HASH_OR_INTERNAL_ID>": {
      "metrics": {
        "prompt_tokens": 100000,
        "compression_saved_tokens": 12000
      },
      "metadata": {
        "key_alias": "production",
        "team_id": "team-123"
      }
    }
  }
}
```

`api_key_breakdown` 可以回答“某个模型、供应商或 Endpoint 的消耗分别来自哪些 Key”。同一个 Key 可能同时出现在多个 breakdown 维度中，因此不要把不同维度下的 Key 数值再次相加，否则会重复统计。需要全局单 Key 总计时，优先使用：

```text
results[].breakdown.api_keys[api_key].metrics
```

或者直接在请求中传入 `api_key` 过滤条件。

### Key 元数据

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `key_alias` | string 或 null | Key 的可读别名 |
| `team_id` | string 或 null | Key 所属团队 ID |

即使 Key 已被删除，接口也会尝试从历史 Key 表补回别名和团队信息。

## 压缩统计口径

### Token 公式

```text
毛节省 Token = 原始输入 Token - 首次压缩后输入 Token

额外输入 Token = 首次请求之后由压缩链路增加的输入消耗

净节省 Token = 毛节省 Token - 额外输入 Token

净节省率 = 净节省 Token / (实际 Prompt Token + 净节省 Token)
```

例如：

```text
毛节省       509,365
额外输入     272,082
净节省       237,283

509,365 - 272,082 = 237,283
```

以下情况会产生额外输入：

- 模型请求 Headroom CCR 召回被压缩的上下文
- LiteLLM 将召回内容加入上下文后再次请求模型
- 压缩结果比原始输入更长

如果额外输入大于毛节省，净节省 Token 和净节省金额会是负数。接口不会把负收益截断为零。

### 金额公式

```text
毛节省金额 = 毛节省 Token × 模型输入单价

额外输入金额 = 额外输入 Token × 模型输入单价

净节省金额 = 净节省 Token × 模型输入单价
```

金额在每次请求仍能确定模型和价格时计算，再写入每日聚合表。不要在跨模型聚合后使用一个统一单价重新估算。

如果模型没有配置 `input_cost_per_token`，Token 统计仍然有效，但三个压缩金额字段会是 `0`。

## Prompt Cache 统计口径

Prompt Cache 数据来自上游模型或供应商返回的 usage，例如：

- `cache_read_input_tokens`
- `cache_creation_input_tokens`
- `cached_tokens`

它不是 AgentOS 或客户端本地缓存开关的命中次数。

缓存节省金额按普通输入价格与缓存读取价格的差额计算：

```text
Prompt Cache 节省金额
= 缓存读取 Token × (普通输入单价 - 缓存读取单价)
```

模型没有独立缓存读取价格时，缓存节省金额为 `0`。

## 首页“成本优化”推荐映射

| 首页指标 | 接口字段 |
| --- | --- |
| Prompt 压缩净节省 Token | `metadata.total_compression_saved_tokens` |
| Prompt 压缩毛节省 Token | `metadata.total_compression_gross_saved_tokens` |
| 召回及二次请求开销 | `metadata.total_compression_extra_input_tokens` |
| Prompt 压缩净节省率 | `metadata.total_compression_net_savings_rate` |
| Prompt 压缩净节省金额 | `metadata.total_compression_savings_spend` |
| Prompt 缓存读取 Token | `metadata.total_cache_read_input_tokens` |
| Prompt 缓存节省金额 | `metadata.total_prompt_caching_savings_spend` |

对外展示压缩效果时应优先使用净节省和净节省率。毛节省适合作为明细，不能单独宣传成最终节省。

## BFF 调用建议

- BFF 使用服务端令牌调用 LiteLLM，浏览器不要直接持有 LiteLLM Master Key
- 原样保留负数，不要对净节省字段执行 `Math.max(value, 0)`
- 百分比展示时，将 `total_compression_net_savings_rate` 乘以 `100`
- 金额为 `0` 时先检查模型价格配置，不要把它解释为没有发生压缩
- Token 字段在数据库中是 BigInt；跨 JSON 边界后仍应避免在前端进行超出 JavaScript 安全整数范围的计算
- 日期筛选必须传用户时区，否则跨 UTC 日期边界时可能与页面日期不一致

TypeScript 示例：

```ts
type CostOptimizationSummary = {
  compressionNetSavedTokens: number;
  compressionGrossSavedTokens: number;
  compressionExtraInputTokens: number;
  compressionNetSavingsRate: number;
  compressionNetSavingsSpend: number;
  cacheReadInputTokens: number;
  cacheSavingsSpend: number;
};

function toCostOptimizationSummary(metadata: Record<string, number>): CostOptimizationSummary {
  return {
    compressionNetSavedTokens: metadata.total_compression_saved_tokens ?? 0,
    compressionGrossSavedTokens: metadata.total_compression_gross_saved_tokens ?? 0,
    compressionExtraInputTokens: metadata.total_compression_extra_input_tokens ?? 0,
    compressionNetSavingsRate: metadata.total_compression_net_savings_rate ?? 0,
    compressionNetSavingsSpend: metadata.total_compression_savings_spend ?? 0,
    cacheReadInputTokens: metadata.total_cache_read_input_tokens ?? 0,
    cacheSavingsSpend: metadata.total_prompt_caching_savings_spend ?? 0,
  };
}
```

生产代码建议使用由 OpenAPI 生成的响应类型，不要长期维护上面的简化手写类型。

## 状态码

| 状态码 | 含义 |
| ---: | --- |
| `200` | 查询成功 |
| `400` | 缺少 `start_date` 或 `end_date`，或请求参数格式不正确 |
| `401` | 未提供有效鉴权令牌 |
| `403` | 非管理员尝试查询其他用户 |
| `500` | 数据库未连接或聚合查询失败 |

## 已知边界

- 这是每日聚合接口，不提供单次请求明细
- 不能只靠该接口识别具体哪次请求发生了 CCR 召回
- 不能精确统计一个未单独落库标识的会话
- 当天数据依赖异步 Spend Log 聚合，刚结束的请求可能短暂延迟出现
- `metadata` 是整个日期范围的总计；`results[].metrics` 是单日总计
- 接口不分页，一次返回所选日期范围内的全部日级数据和 breakdown
- 如果需要请求级、会话级或召回事件级审计，应查询 `LiteLLM_SpendLogs` 或提供独立的明细接口
