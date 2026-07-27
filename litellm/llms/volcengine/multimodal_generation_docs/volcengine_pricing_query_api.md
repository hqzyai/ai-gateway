# 火山引擎模型计费规则查询接口文档

文档版本：1.0

数据快照日期：2026-07-27

适用范围：`model_prices_and_context_window.json` 中 `litellm_provider` 为 `volcengine` 的全部模型

当前适配数量：23 个模型条目

## 1. 文档目的

本文用于约定外部系统如何读取 LiteLLM 的模型价格 JSON，并返回指定火山引擎模型的计费规则

本文只描述价格规则查询和字段语义，不把 JSON 中的价格当作火山引擎实时账单。实际结算仍应以火山引擎账单为准

## 2. 数据源

主价格表：

```text
model_prices_and_context_window.json
```

LiteLLM 本地备份：

```text
litellm/model_prices_and_context_window_backup.json
```

两份文件中的火山引擎条目应保持一致。对接方读取仓库文件时，以根目录的 `model_prices_and_context_window.json` 为主

筛选条件：

```text
entry.litellm_provider == "volcengine"
```

JSON 标准不支持注释。当前文件使用以下两个字段承载价格说明：

- `source`：价格来源页面
- `metadata.notes`：价格口径、原始标价或模型说明

## 3. 现有 LiteLLM 查询接口

### 3.1 获取完整价格表

```http
GET /public/litellm_model_cost_map
```

示例：

```bash
curl -s "http://localhost:4000/public/litellm_model_cost_map"
```

当前接口是公开路由，不要求 `Authorization`。响应没有 `data` 包装层，顶层 Key 就是模型 Key

该接口返回 Proxy 当前内存中实际加载的 `litellm.model_cost`。如果 Proxy 配置了远程价格表、运行时重载或 deployment 自定义价格，接口结果可能与仓库静态 JSON 快照不同

截取指定模型：

```bash
curl -s "http://localhost:4000/public/litellm_model_cost_map" \
  | jq '.["volcengine/doubao-seed-2-0-pro-260215"]'
```

响应示意：

```json
{
  "litellm_provider": "volcengine",
  "mode": "chat",
  "input_cost_per_token": 0.0000032,
  "output_cost_per_token": 0.000016,
  "cache_read_input_token_cost": 0.00000064,
  "input_cost_per_token_above_32768_tokens": 0.0000048,
  "output_cost_per_token_above_32768_tokens": 0.000024,
  "cache_read_input_token_cost_above_32768_tokens": 0.00000096,
  "input_cost_per_token_above_131072_tokens": 0.0000096,
  "output_cost_per_token_above_131072_tokens": 0.000048,
  "cache_read_input_token_cost_above_131072_tokens": 0.00000192
}
```

该接口当前不支持用 Query 参数只查询一个模型。调用方需要从完整响应中按模型 Key 提取

### 3.2 获取 Proxy 已配置模型

```http
GET /model/info
Authorization: Bearer <API_KEY>
```

该接口返回当前 Proxy 已配置且调用方有权访问的 deployment。价格位于每个 deployment 的 `model_info` 中，自定义 deployment 价格优先，缺失字段才由全局价格表补齐

如果对接方需要的是火山引擎的完整标准价格规则，应读取完整价格表；如果需要的是当前 Proxy 部署后的最终价格，应读取 `/model/info`

## 4. 建议的对接接口

如果对接方需要封装一个“按模型查询”的接口，建议使用以下契约。该路由是对接建议，不是 LiteLLM 当前内置路由

```http
GET /v1/model-pricing/{model_key}
```

成功响应：

```json
{
  "model": "volcengine/doubao-seed-2-0-pro-260215",
  "provider": "volcengine",
  "mode": "chat",
  "currency": "CNY",
  "pricing": {
    "input_cost_per_token": 0.0000032,
    "output_cost_per_token": 0.000016,
    "cache_read_input_token_cost": 0.00000064,
    "input_cost_per_token_above_32768_tokens": 0.0000048,
    "output_cost_per_token_above_32768_tokens": 0.000024,
    "cache_read_input_token_cost_above_32768_tokens": 0.00000096,
    "input_cost_per_token_above_131072_tokens": 0.0000096,
    "output_cost_per_token_above_131072_tokens": 0.000048,
    "cache_read_input_token_cost_above_131072_tokens": 0.00000192
  },
  "source": "https://www.volcengine.com/docs/82379/1330310",
  "notes": "CN list prices recorded 1:1 as USD. Ark bills a request at one input-length tier: prompt <=32768, <=131072, else <=262144 tokens"
}
```

模型不存在：

```http
HTTP/1.1 404 Not Found
Content-Type: application/json
```

```json
{
  "error": {
    "code": "MODEL_PRICING_NOT_FOUND",
    "message": "No pricing rule found for model: volcengine/unknown-model"
  }
}
```

价格未配置和价格为零必须区分：

- 字段不存在：价格未知或该计费维度不适用
- 字段存在且等于 `0.0`：JSON 明确配置为零

不要把不存在的价格字段自动补成 `0.0` 返回，否则调用方无法区分“免费”和“未配置”

## 5. 模型 Key 匹配规则

当前条目同时存在新旧两种命名方式：

- 新条目：`volcengine/<model-id>`
- 历史条目：没有 `volcengine/` 前缀，例如 `doubao-embedding`

按模型查询时建议按以下顺序匹配：

1. 使用请求值做完全匹配
2. 请求值没有 `/` 时，再尝试 `volcengine/<请求值>`
3. 都未命中时返回 404

模型 Key 区分大小写，不建议使用模糊匹配或 `contains` 匹配

伪代码：

```text
rule = price_map.get(requested_model)

if rule is null and "/" not in requested_model:
    rule = price_map.get("volcengine/" + requested_model)

if rule is null or rule.litellm_provider != "volcengine":
    return 404
```

## 6. 币种和数值口径

当前非零火山价格采用“人民币标价数字 1:1 写入 LiteLLM 美元价格字段”的仓库约定

例如火山标价为 3.2 元/百万 token，JSON 中写为：

```json
{
  "input_cost_per_token": 0.0000032,
  "metadata": {
    "notes": "CN list prices recorded 1:1 as USD"
  }
}
```

换算过程：

```text
3.2 元 / 1,000,000 token = 0.0000032 / token
```

这里没有进行人民币到美元的汇率换算。对接接口应明确返回 `currency: "CNY"`，或者原样返回 `metadata.notes`，避免调用方把数值误认为真实美元金额

建议使用 Decimal 类型解析价格。科学计数法和普通小数表示等价，例如 `7e-7` 等于 `0.0000007`

## 7. 通用字段说明

### 7.1 标识与说明字段

| 字段 | 类型 | 含义 | 是否参与计费 |
|---|---|---|---|
| 顶层模型 Key | string | 模型的唯一价格表标识，例如 `volcengine/doubao-seedance-2-0-260128` | 用于查找规则 |
| `litellm_provider` | string | LiteLLM Provider，火山引擎固定为 `volcengine` | 用于选择 Provider 计费实现 |
| `mode` | string | 模型业务类型，决定使用哪组价格字段和公式 | 是 |
| `source` | string | 官方价格或模型信息来源 URL | 否 |
| `metadata.notes` | string | 人工可读的价格口径、原始标价、档位或模型说明 | 否 |
| `supported_endpoints` | string[] | 当前模型预期支持的 API 路径 | 否，主要用于展示和接入说明 |

### 7.2 容量和输出字段

| 字段 | 类型 | 含义 | 是否参与计费 |
|---|---|---|---|
| `max_input_tokens` | integer | 最大输入 token 数 | 不直接参与，仅用于容量校验和展示 |
| `max_output_tokens` | integer | 最大输出 token 数 | 不直接参与 |
| `max_tokens` | integer | 兼容字段，通常表示最大生成 token 数 | 不直接参与 |
| `output_vector_size` | integer | Embedding 默认或最大向量维度 | 否 |

### 7.3 能力字段

| 字段 | 含义 |
|---|---|
| `supports_function_calling` | 是否支持函数或工具调用 |
| `supports_tool_choice` | 是否支持由请求显式指定工具选择策略 |
| `supports_reasoning` | 是否支持推理能力 |
| `supports_vision` | Chat 模型是否支持图片视觉输入 |
| `supports_prompt_caching` | 是否支持 Prompt 缓存 |
| `supports_assistant_prefill` | 是否支持 Assistant Prefill |
| `supports_embedding_image_input` | Embedding 是否支持图片输入 |
| `supports_video_input` | 是否支持视频输入 |
| `supports_audio_input` | 是否支持音频输入 |
| `supports_audio_output` | 是否支持音频输出 |
| `supported_modalities` | 支持的输入模态列表 |
| `supported_output_modalities` | 支持的输出模态列表 |

这些字段用于能力发现，不参与金额计算

## 8. 计费字段字典

| 字段 | 单位 | 含义 |
|---|---|---|
| `input_cost_per_token` | 货币/token | 普通输入文本 token 单价 |
| `output_cost_per_token` | 货币/token | 普通输出 token 单价 |
| `cache_read_input_token_cost` | 货币/token | 命中上下文缓存的输入 token 单价 |
| `input_cost_per_token_above_{N}_tokens` | 货币/token | 当本次 `prompt_tokens > N` 时使用的整单输入单价 |
| `output_cost_per_token_above_{N}_tokens` | 货币/token | 当本次 `prompt_tokens > N` 时使用的整单输出单价 |
| `cache_read_input_token_cost_above_{N}_tokens` | 货币/token | 当本次 `prompt_tokens > N` 时使用的整单缓存命中单价 |
| `tiered_pricing` | object[] | 输入长度档位的预算预留表示，当前火山实际 spend 计算仍以 `_above_` 字段为准 |
| `input_cost_per_image_token` | 货币/token | Embedding 图片 token 单价；视频帧 token 当前也按该价格 |
| `input_cost_per_character` | 货币/字符 | TTS 输入字符单价 |
| `input_cost_per_second` | 货币/秒 | ASR 输入音频时长单价 |
| `input_cost_per_image` | 货币/张 | 图片生成或图片编辑的输入图片单价 |
| `output_cost_per_image` | 货币/张 | 普通输出图片单价 |
| `output_cost_per_image_above_{N}_tokens` | 货币/张 | 当 `usage.output_tokens > N` 时的整张输出图片单价 |
| `video_token_pricing` | object | 按是否有视频输入和分辨率选择的视频生成 token 单价表 |

`{N}` 是完整 token 数。当前火山档位使用 `32768`、`131072` 和 `16384`，不能写成 `32k`、`128k` 或 `16k`，因为 `k` 在 LiteLLM 中按 1000 解析，不是按 1024 解析

## 9. 按 mode 选择计费规则

| `mode` | 当前模型数 | 主要计费维度 | 关键字段 |
|---|---:|---|---|
| `chat` | 7 | 输入、输出、缓存 token | `input_cost_per_token`、`output_cost_per_token`、`cache_read_input_token_cost`、`*_above_*` |
| `embedding` | 7 | 文本 token、图片/视频帧 token | `input_cost_per_token`、`input_cost_per_image_token` |
| `audio_speech` | 1 | 输入字符数 | `input_cost_per_character` |
| `audio_transcription` | 1 | 输入音频秒数 | `input_cost_per_second` |
| `image_generation` | 4 | 输入图片张数、输出图片张数 | `input_cost_per_image`、`output_cost_per_image`、`output_cost_per_image_above_*` |
| `video_generation` | 3 | 生成 token、是否有视频输入、分辨率 | `video_token_pricing` |

### 9.1 Chat

无缓存时：

```text
cost = prompt_tokens × input_rate
     + completion_tokens × output_rate
```

存在缓存命中明细时：

```text
cost = non_cached_text_tokens × input_rate
     + cache_hit_tokens × cache_read_rate
     + completion_tokens × output_rate
```

档位由整次请求的 `prompt_tokens` 决定，不是累进计费：

- `prompt_tokens <= 32768`：基础档
- `32768 < prompt_tokens <= 131072`：`above_32768` 档
- `131072 < prompt_tokens <= 262144`：`above_131072` 档

命中高档后，整次请求的输入、输出和缓存 token 都使用该档单价

`tiered_pricing` 当前用于 Proxy 请求前的预算预留；火山实际 spend 计算读取顶层的基础字段和 `_above_` 字段。对接方如果自己计算实际费用，也应读取 `_above_` 字段

### 9.2 Embedding

纯文本：

```text
cost = text_tokens × input_cost_per_token
```

多模态：

```text
cost = text_tokens × input_cost_per_token
     + image_tokens × input_cost_per_image_token
     + video_frame_tokens × input_cost_per_image_token
```

### 9.3 TTS

```text
cost = input_character_count × input_cost_per_character
```

当前价格是 3 元/万字符，对应 `0.0003`/字符

### 9.4 ASR

```text
cost = audio_duration_seconds × input_cost_per_second
```

当前价格是 4.5 元/小时，对应 `4.5 / 3600 = 0.00125`/秒

### 9.5 图片生成和图片编辑

```text
cost = input_image_count × input_cost_per_image
     + generated_image_count × selected_output_cost_per_image
```

输出图片单价选择：

```text
usage.output_tokens > N
    ? output_cost_per_image_above_N_tokens
    : output_cost_per_image
```

边界是严格大于。例如 `usage.output_tokens == 16384` 使用普通档，`16385` 才使用 `above_16384` 档

### 9.6 视频生成

```text
cost = usage.completion_tokens × selected_video_token_rate
```

档位选择顺序：

1. 请求含视频输入时使用 `video_input`，否则使用 `no_video_input`
2. 如果存在分辨率专属键，优先使用 `<input_type>_<resolution>`
3. 没有分辨率专属键时，回落到不带分辨率的基础键

例如无视频输入的 1080p 请求优先读取 `no_video_input_1080p`；Fast 和 Mini 没有分辨率专属键，因此所有分辨率都回落到 `no_video_input` 或 `video_input`

## 10. 当前全部模型计费规则

下表中的 token 价格为了便于阅读，统一展示为“元/百万 token”。JSON 实际保存的是每 token 单价

### 10.1 Chat：零价或未维护真实价格的条目

| 模型 Key | 输入 | 输出 | 最大输入 | 最大输出 | 说明 |
|---|---:|---:|---:|---:|---|
| `deepseek-v3-2-251201` | 0 | 0 | 98,304 | 32,768 | 当前 JSON 明确为 0；不能据此推断火山官方永久免费 |
| `glm-4-7-251222` | 0 | 0 | 204,800 | 131,072 | 当前 JSON 明确为 0；不能据此推断火山官方永久免费 |
| `kimi-k2-thinking-251104` | 0 | 0 | 229,376 | 32,768 | 当前 JSON 明确为 0；不能据此推断火山官方永久免费 |

### 10.2 Chat：Doubao Seed 2.0 分档价格

单位：元/百万 token

| 模型 Key | 输入 token 范围 | 输入 | 缓存命中 | 输出 |
|---|---:|---:|---:|---:|
| `volcengine/doubao-seed-2-0-pro-260215` | 0 - 32,768 | 3.20 | 0.64 | 16.00 |
| 同上 | 32,769 - 131,072 | 4.80 | 0.96 | 24.00 |
| 同上 | 131,073 - 262,144 | 9.60 | 1.92 | 48.00 |
| `volcengine/doubao-seed-2-0-lite-260215` | 0 - 32,768 | 0.60 | 0.12 | 3.60 |
| 同上 | 32,769 - 131,072 | 0.90 | 0.18 | 5.40 |
| 同上 | 131,073 - 262,144 | 1.80 | 0.36 | 10.80 |
| `volcengine/doubao-seed-2-0-mini-260215` | 0 - 32,768 | 0.20 | 0.04 | 2.00 |
| 同上 | 32,769 - 131,072 | 0.40 | 0.08 | 4.00 |
| 同上 | 131,073 - 262,144 | 0.80 | 0.16 | 8.00 |
| `volcengine/doubao-seed-2-0-code-preview-260215` | 0 - 32,768 | 3.20 | 0.64 | 16.00 |
| 同上 | 32,769 - 131,072 | 4.80 | 0.96 | 24.00 |
| 同上 | 131,073 - 262,144 | 9.60 | 1.92 | 48.00 |

这 4 个模型的最大输出均为 131,072 token

### 10.3 Embedding

| 模型 Key | 文本价格 | 图片/视频帧价格 | 向量维度 | 说明 |
|---|---:|---:|---:|---|
| `doubao-embedding` | 0 | 不适用 | 2,560 | 历史零价条目 |
| `doubao-embedding-large` | 0 | 不适用 | 2,048 | 历史零价条目 |
| `doubao-embedding-large-text-240915` | 0 | 不适用 | 4,096 | 历史零价条目 |
| `doubao-embedding-large-text-250515` | 0 | 不适用 | 2,048 | 历史零价条目 |
| `doubao-embedding-text-240715` | 0 | 不适用 | 2,560 | 历史零价条目 |
| `volcengine/doubao-embedding-vision-250615` | 0.70 元/百万 token | 1.80 元/百万 token | 2,048 | 图片和视频帧 token 使用图片 token 价格 |
| `volcengine/doubao-embedding-vision-251215` | 0.70 元/百万 token | 1.80 元/百万 token | 2,048 | 图片和视频帧 token 使用图片 token 价格 |

前 5 个模型的 `0.0` 是当前 JSON 值，表示本地价格表没有维护非零真实价格，不应对外宣称为火山官方免费模型

### 10.4 TTS 和 ASR

| 模型 Key | `mode` | 计费规则 | JSON 单价 |
|---|---|---|---:|
| `volcengine/seed-tts-2.0` | `audio_speech` | 输入字符数 × 每字符单价 | `input_cost_per_character = 0.0003` |
| `volcengine/volc.bigasr.auc_turbo` | `audio_transcription` | 输入音频秒数 × 每秒单价 | `input_cost_per_second = 0.00125` |

### 10.5 图片生成和图片编辑

单位：元/张

| 模型 Key | 输入图 | 普通输出图 | 大图输出 | 大图条件 |
|---|---:|---:|---:|---|
| `volcengine/doubao-seedream-5-0-260128` | 0.00 | 0.22 | 不适用 | 无大图档 |
| `volcengine/doubao-seedream-5-0-pro-260628` | 0.02 | 0.30 | 0.60 | `usage.output_tokens > 16384` |
| `volcengine/doubao-seedream-4-0-250828` | 0.00 | 0.20 | 不适用 | 无大图档 |
| `volcengine/doubao-seedream-4-5-251128` | 0.00 | 0.25 | 不适用 | 无大图档 |

文生图时输入图数量为 0。图生图和图片编辑会按实际输入图片数量计算输入图费用

### 10.6 视频生成

单位：元/百万 `completion_tokens`

| 模型 Key | 无视频输入 | 无视频输入 1080p | 无视频输入 4K | 有视频输入 | 有视频输入 1080p | 有视频输入 4K |
|---|---:|---:|---:|---:|---:|---:|
| `volcengine/doubao-seedance-2-0-260128` | 46 | 51 | 26 | 28 | 31 | 16 |
| `volcengine/doubao-seedance-2-0-fast-260128` | 37 | 回落到 37 | 回落到 37 | 22 | 回落到 22 | 回落到 22 |
| `volcengine/doubao-seedance-2-0-mini-260615` | 23 | 回落到 23 | 回落到 23 | 14 | 回落到 14 | 回落到 14 |

对应 JSON 每 token 单价：

```json
{
  "volcengine/doubao-seedance-2-0-260128": {
    "video_token_pricing": {
      "no_video_input": 0.000046,
      "no_video_input_1080p": 0.000051,
      "no_video_input_4k": 0.000026,
      "video_input": 0.000028,
      "video_input_1080p": 0.000031,
      "video_input_4k": 0.000016
    }
  },
  "volcengine/doubao-seedance-2-0-fast-260128": {
    "video_token_pricing": {
      "no_video_input": 0.000037,
      "video_input": 0.000022
    }
  },
  "volcengine/doubao-seedance-2-0-mini-260615": {
    "video_token_pricing": {
      "no_video_input": 0.000023,
      "video_input": 0.000014
    }
  }
}
```

## 11. `metadata.notes` 注释解释

| 注释内容 | 对接含义 |
|---|---|
| `CN list prices recorded 1:1 as USD` | 原始价格来自中国区人民币标价，数值未做汇率换算，直接写入 LiteLLM 的通用价格字段 |
| `CN list price is CNY ...` | 注释中记录了火山官方人类可读单价，以及转换为 JSON 单位前的价格 |
| `Ark bills a request at one input-length tier...` | Chat 按本次请求总输入长度选择一个档位，整单采用该档，不是累进计价 |
| `...dimensions` | Embedding 向量维度说明，不是价格字段 |
| `video frame tokens use the image rate` | 多模态 Embedding 的视频帧 token 按 `input_cost_per_image_token` 计费 |
| 没有 `metadata.notes` | 当前条目没有补充说明；不能据此推断币种、价格来源或免费政策 |

`metadata.notes` 只用于说明和展示，计算程序不能解析自然语言注释来决定价格。所有可执行规则必须来自结构化价格字段

## 12. 完整响应示例

### 12.1 Chat 分档模型

```json
{
  "model": "volcengine/doubao-seed-2-0-lite-260215",
  "provider": "volcengine",
  "mode": "chat",
  "currency": "CNY",
  "pricing": {
    "input_cost_per_token": 0.0000006,
    "output_cost_per_token": 0.0000036,
    "cache_read_input_token_cost": 0.00000012,
    "input_cost_per_token_above_32768_tokens": 0.0000009,
    "output_cost_per_token_above_32768_tokens": 0.0000054,
    "cache_read_input_token_cost_above_32768_tokens": 0.00000018,
    "input_cost_per_token_above_131072_tokens": 0.0000018,
    "output_cost_per_token_above_131072_tokens": 0.0000108,
    "cache_read_input_token_cost_above_131072_tokens": 0.00000036
  }
}
```

### 12.2 图片模型

```json
{
  "model": "volcengine/doubao-seedream-5-0-pro-260628",
  "provider": "volcengine",
  "mode": "image_generation",
  "currency": "CNY",
  "pricing": {
    "input_cost_per_image": 0.02,
    "output_cost_per_image": 0.3,
    "output_cost_per_image_above_16384_tokens": 0.6
  }
}
```

### 12.3 多模态 Embedding

```json
{
  "model": "volcengine/doubao-embedding-vision-251215",
  "provider": "volcengine",
  "mode": "embedding",
  "currency": "CNY",
  "pricing": {
    "input_cost_per_token": 0.0000007,
    "input_cost_per_image_token": 0.0000018,
    "output_cost_per_token": 0.0
  }
}
```

## 13. 兼容性要求

对接实现应满足以下要求：

1. 保留未知字段，避免 LiteLLM 后续新增价格维度时被旧代码丢弃
2. 使用 `mode` 选择计费字段，不能假定所有模型都按 token 计费
3. 区分字段缺失、`null` 和数值 `0.0`
4. 阈值判断必须使用严格大于 `>`
5. Chat 档位是整单档位，不是累进档位
6. 视频分辨率专属价优先，缺失时必须回落到基础价
7. 不要从 `metadata.notes` 解析可执行规则
8. 返回原始每单位单价时不得擅自乘以一百万；如需展示“每百万 token”，应额外提供展示字段
9. 价格更新后应重新加载数据，不建议永久缓存进程启动时的副本
10. 未命中模型或关键计费字段缺失时应返回明确错误或“未配置”状态，不能静默按零价处理

## 14. 对接验收样例

### 14.1 Chat 档位边界

使用 `volcengine/doubao-seed-2-0-pro-260215`：

| `prompt_tokens` | 期望输入单价 | 期望输出单价 |
|---:|---:|---:|
| 32,768 | 0.0000032 | 0.000016 |
| 32,769 | 0.0000048 | 0.000024 |
| 131,072 | 0.0000048 | 0.000024 |
| 131,073 | 0.0000096 | 0.000048 |

### 14.2 图片档位边界

使用 `volcengine/doubao-seedream-5-0-pro-260628`：

| 输入图 | 输出图 | `output_tokens` | 期望费用 |
|---:|---:|---:|---:|
| 0 | 1 | 16,384 | 0.30 |
| 0 | 1 | 16,385 | 0.60 |
| 2 | 1 | 16,384 | 0.34 |
| 1 | 1 | 18,000 | 0.62 |

### 14.3 视频档位和回落

| 模型 | 视频输入 | 分辨率 | 期望每 token 单价 |
|---|---|---|---:|
| Seedance 2.0 | 否 | 1080p | 0.000051 |
| Seedance 2.0 | 是 | 4K | 0.000016 |
| Seedance 2.0 Fast | 否 | 1080p | 0.000037 |
| Seedance 2.0 Mini | 是 | 720p | 0.000014 |

## 15. 当前已知限制

- `tiered_pricing` 不直接参与火山实际 spend 计算，必须同时保留对应的 `_above_` 字段
- 3 个 Chat 条目和 5 个历史文本 Embedding 条目当前价格为 `0.0`，但没有证据表明它们是火山官方永久免费模型
- 当前币种没有独立的结构化 `currency` 字段，只通过 `metadata.notes` 描述
- `/public/litellm_model_cost_map` 返回完整价格表，当前没有服务端单模型过滤参数
- `supported_endpoints` 和能力字段主要是元数据，不代表价格查询接口会验证实际路由可用性
