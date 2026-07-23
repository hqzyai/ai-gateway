# 火山引擎（Ark）模型接入与计费配置指南

适用范围：`litellm_provider: volcengine` 的所有条目，覆盖 chat（含 `/v1/responses`）、文生图 / 图生图 / 图片编辑、视频生成、embedding

代码位置：`litellm/llms/volcengine/`

价格表：`model_prices_and_context_window.json`（源）与 `litellm/model_prices_and_context_window_backup.json`（备份，必须与源逐字节一致）

本文所有结论都在当前分支上跑代码验证过，验证命令见第 9 节

---

## 1. 速查表

| 业务类型 | `mode` | 计费单位 | 关键价格字段 | 实际算钱的代码 |
|---|---|---|---|---|
| Chat / Responses | `chat` | token | `input_cost_per_token`、`output_cost_per_token`、`input_cost_per_token_above_{N}k_tokens` | `litellm/litellm_core_utils/llm_cost_calc/utils.py` 的 `generic_cost_per_token` |
| 文生图 / 图生图 / 图片编辑 | `image_generation` | 张 | `input_cost_per_image`、`output_cost_per_image`、`output_cost_per_image_above_16384_tokens` | `litellm/llms/volcengine/image_generation/cost_calculator.py` |
| 视频生成 | `video_generation` | token | `video_token_pricing` | `litellm/llms/openai/cost_calculation.py` 的 `video_generation_cost` |
| Embedding | `embedding` | token | `input_cost_per_token` | `generic_cost_per_token` |

重要：`tiered_pricing` 这个字段对火山引擎**不生效**，写了等于没写，详见第 3.3 节

---

## 2. 条目骨架与通用约定

### 2.1 Key 命名

新条目一律写成 `volcengine/<火山模型 ID>`，例如 `volcengine/doubao-seedream-5-0-260128`

`get_model_info` 会自动尝试补 `volcengine/` 前缀，所以调用方传 `doubao-seedream-5-0-260128` 也能命中同一条目。历史遗留的 embedding 条目（`doubao-embedding` 等）没有前缀，属于旧写法，新增时不要模仿

### 2.2 每条都要有的字段

```json
{
    "litellm_provider": "volcengine",
    "mode": "<chat | image_generation | video_generation | embedding>",
    "source": "https://www.volcengine.com/docs/82379/1544106",
    "metadata": {
        "notes": "CN list prices recorded 1:1 as USD"
    }
}
```

`source` 指向火山官方定价页或文档；`metadata.notes` 用来说明币种换算口径

### 2.3 币种口径

本仓库对火山引擎采用**人民币数字 1:1 记为美元数字**：火山标价 0.22 元/张就写 `0.22`，不做汇率换算。新增或改价时保持同一口径，并在 `metadata.notes` 里写清楚，否则对账会乱

### 2.4 `supported_endpoints`

对上述四种模式，这个字段目前只是元数据（给 UI 和文档看），并不会拦截路由。仍然建议按实际填：

- chat：`["/v1/chat/completions"]`，若走 Responses 桥接再加 `"/v1/responses"`
- 图片：`["/v1/images/generations"]`
- 视频：`["/v1/videos"]`

### 2.5 两个价格表必须同步

`model_prices_and_context_window.json` 和 `litellm/model_prices_and_context_window_backup.json` 的火山条目当前完全一致，改一个必须改另一个。`tests/test_litellm/llms/volcengine/videos/test_volcengine_video_transformation.py::test_seedance_model_is_registered_for_video_generation` 会同时读这两个文件做断言

---

## 3. Chat（含 Responses）

### 3.1 平价（最简单）

```json
"volcengine/<model>": {
    "litellm_provider": "volcengine",
    "mode": "chat",
    "max_input_tokens": 262144,
    "max_output_tokens": 131072,
    "max_tokens": 131072,
    "input_cost_per_token": 3.2e-06,
    "output_cost_per_token": 1.6e-05
}
```

单位是「每 token 单价」。Ark 的 `ark_get_model` 给的是「元 / 千 token」，除以 1000：`0.0032` 元/千 token -> `3.2e-06`。官网页面上写的是「元 / 百万 token」，那就除以 1000000

### 3.2 按输入长度分档（火山 doubao 的主流计费方式）

火山的 doubao 系列按**单次请求的输入长度**落档，整个请求（输入和输出）都按该档单价计费，不是累进制。LiteLLM 里对应的就是 `_above_{N}_tokens` 系列字段：

```json
"volcengine/doubao-seed-2-0-pro-260215": {
    "litellm_provider": "volcengine",
    "mode": "chat",
    "max_input_tokens": 262144,
    "max_output_tokens": 131072,

    "input_cost_per_token": 3.2e-06,
    "output_cost_per_token": 1.6e-05,
    "cache_read_input_token_cost": 6.4e-07,

    "input_cost_per_token_above_32768_tokens": 4.8e-06,
    "output_cost_per_token_above_32768_tokens": 2.4e-05,
    "cache_read_input_token_cost_above_32768_tokens": 9.6e-07,

    "input_cost_per_token_above_131072_tokens": 9.6e-06,
    "output_cost_per_token_above_131072_tokens": 4.8e-05,
    "cache_read_input_token_cost_above_131072_tokens": 1.92e-06
}
```

阈值一定要用火山 `billing_condition.max_prompt_tokens` 的原值（32768、131072），别写成 `32k` / `128k`——那会被解析成 32000 / 128000，和火山的实际边界差 768 个 token，落在这个区间的请求会算错档

语义（见 `_get_token_base_cost`，`litellm/litellm_core_utils/llm_cost_calc/utils.py:201`）：

1. 扫描所有以 `input_cost_per_token_above_` 开头的键，按阈值从大到小排序
2. 取第一个满足 `usage.prompt_tokens > 阈值` 的档
3. 该档的 input / output / cache 单价套用到**整个请求的全部 token**
4. 没有任何档命中，就用基础的 `input_cost_per_token` / `output_cost_per_token`

阈值写法支持 `_above_32768_tokens`（= 32768）和 `_above_32k_tokens`（= 32 × 1000 = 32000）两种。火山的档位边界是 2 的幂，所以**必须写完整数字**，用 `k` 会差 768 个 token

边界是严格大于：`prompt_tokens == 32768` 落在基础档，正好对上火山 `billing_condition` 的 `max_prompt_tokens: 32768`（描述写作「输入小于≤32k」）

Pro / Code Preview 的实测（`completion_tokens = 1000`）：

| prompt_tokens | 命中档 | input 单价 | output 单价 |
|---:|---|---:|---:|
| 32,768 | 基础档 | 3.2e-06 | 1.6e-05 |
| 32,769 | above_32768 | 4.8e-06 | 2.4e-05 |
| 131,072 | above_32768 | 4.8e-06 | 2.4e-05 |
| 131,073 | above_131072 | 9.6e-06 | 4.8e-05 |

同一套 `_above_` 后缀也适用于缓存字段：`cache_read_input_token_cost_above_32768_tokens`、`cache_creation_input_token_cost_above_32768_tokens`。缓存档由 input 档带出来，选中哪个 input 档就用同后缀的缓存键

### 3.3 `tiered_pricing` 对火山引擎不算钱

`tiered_pricing` 只有 **dashscope** 有专门的 cost calculator 去读（`litellm/llms/dashscope/cost_calculator.py`）。`cost_per_token` 的 provider 分发里没有 volcengine 分支，火山走的是通用的 `generic_cost_per_token`，而通用路径**完全不读 `tiered_pricing`**

历史上 4 个 `volcengine/doubao-seed-2-0-*` 条目只配了 `tiered_pricing`，没有 `input_cost_per_token` / `output_cost_per_token`，实际扣费一直是 0。现已改为 3.2 节的 `_above_` 字段

`tiered_pricing` 仍然保留，因为 proxy 的预扣预算只认它（`litellm/proxy/spend_tracking/budget_reservation.py:973`），删掉会让长上下文请求的预留额度按基础档算、少留最多 3 倍。两套表示必须写一致的值：

- `_above_` 字段决定**实际落库的 spend**
- `tiered_pricing` 决定**请求前的预算预留**

两者的档位边界语义本来就一致（`select_tier_for_input` 用 `range_start < input_tokens <= range_end`，`_get_token_base_cost` 用 `prompt_tokens > threshold`），所以同一个请求两边选中同一档。`tests/test_litellm/llms/volcengine/test_volcengine.py::test_seed_2_0_tiered_pricing_agrees_with_above_threshold_fields` 会断言两边逐值相等，防止改价时只改一半

### 3.4 能力开关

和计费无关但要一起配：

```json
"supports_function_calling": true,
"supports_tool_choice": false,
"supports_vision": true,
"supports_reasoning": true,
"supports_prompt_caching": true,
"supports_assistant_prefill": true
```

---

## 4. 文生图 / 图生图 / 图片编辑

三种场景共用一条价格条目和一个 cost calculator。`/v1/images/edits` 在 `route_image_generation_cost_calculator` 里按 provider 分发，火山的 image edit 和 image generation 走的是同一个函数（`litellm/litellm_core_utils/llm_cost_calc/utils.py:1193`），而且 `VolcEngineImageEditConfig` 底层打的也是 `/images/generations`

### 4.1 计费公式

```
费用 = input_cost_per_image  × usage.input_images
     + 输出单价              × usage.generated_images

输出单价 = output_cost_per_image_above_16384_tokens   若 usage.output_tokens > 16384 且该字段存在
         = output_cost_per_image                      其他情况
```

`input_images` / `generated_images` / `output_tokens` 全部取自火山响应的 `usage`。火山没返回 `input_images` 时，回落到请求体里 `image` 字段的数量；没返回 `generated_images` 时回落到 `len(data)`

### 4.2 单档模型（Seedream 5.0 Lite）

```json
"volcengine/doubao-seedream-5-0-260128": {
    "litellm_provider": "volcengine",
    "mode": "image_generation",
    "input_cost_per_image": 0.0,
    "output_cost_per_image": 0.22,
    "supported_endpoints": ["/v1/images/generations"],
    "source": "https://console.volcengine.com/ark/region:cn-beijing/model?view=DEFAULT_VIEW&groupType=ModelGroups",
    "metadata": {"notes": "CN list price is CNY 0.22 per output image; recorded 1:1 as USD"}
}
```

输入图不额外收费就写 `"input_cost_per_image": 0.0`，不要省略，省略会让人以为漏配了

### 4.3 双档模型（Seedream 5.0 Pro，带大图档）

```json
"volcengine/doubao-seedream-5-0-pro-260628": {
    "litellm_provider": "volcengine",
    "mode": "image_generation",
    "input_cost_per_image": 0.02,
    "output_cost_per_image": 0.3,
    "output_cost_per_image_above_16384_tokens": 0.6,
    "supported_endpoints": ["/v1/images/generations"],
    "metadata": {"notes": "CN list prices are CNY 0.02 per input image, CNY 0.30 per standard output image, and CNY 0.60 per output image above 16384 output tokens; recorded 1:1 as USD"}
}
```

实测：

| 场景 | input_images | generated_images | output_tokens | 费用 |
|---|---:|---:|---:|---:|
| Lite 文生图 | 0 | 1 | 16384 | 0.22 |
| Pro 普通文生图 | 0 | 1 | 16384 | 0.30 |
| Pro 图生图（2 张输入） | 2 | 1 | 16384 | 0.34 |
| Pro 大图图生图 | 1 | 1 | 18000 | 0.62 |

### 4.4 任意阈值、任意档数

阈值不限于 16384，也不限于两档。写几个 `output_cost_per_image_above_{N}_tokens` 就是几档，`N` 支持 `8192` 和 `8k` 两种写法：

```json
"input_cost_per_image": 0.02,
"output_cost_per_image": 0.3,
"output_cost_per_image_above_4096_tokens": 0.45,
"output_cost_per_image_above_8192_tokens": 0.6
```

选档规则和 chat 的 `_above_` 一致（`select_above_threshold_rate`，`litellm/litellm_core_utils/llm_cost_calc/utils.py`）：取 `output_tokens` 越过的**最高**那一档，都没越过就用 `output_cost_per_image`。边界是严格大于，`output_tokens == 4096` 仍走基础档

上面这份配置的实际结果：

| output_tokens | 命中档 | 输出单价 |
|---:|---|---:|
| 4096 | 基础 | 0.30 |
| 4097 | above_4096 | 0.45 |
| 8192 | above_4096 | 0.45 |
| 8193 | above_8192 | 0.60 |

### 4.5 对火山不适用的字段

以下字段在别的 provider 上有用，但火山走的是自己的 calculator，写了也不会被读：

- DALL-E 风格的 `<size>/<quality>` 嵌套价格表
- `output_cost_per_image_token`（通用 token 计价，火山按张计）
- `input_cost_per_pixel` / `output_cost_per_pixel`

---

## 5. 视频生成

### 5.1 计费公式

```
费用 = video_token_pricing[档位] × usage.completion_tokens
```

火山的视频任务查询响应里 `usage.completion_tokens` 就是官方对账口径，且 `total_tokens == completion_tokens`

### 5.2 档位键名规则

`_video_output_cost_per_token`（`litellm/llms/openai/cost_calculation.py`）按两个维度选键：

```
基础键 = "video_input"     若请求带视频输入
       = "no_video_input"  否则

最终键 = f"{基础键}_{resolution}"  若该键存在于 video_token_pricing
       = 基础键                    否则（回落）
```

分辨率后缀**没有白名单**，火山报什么分辨率就找什么键。想给 720p 单独定价，直接加 `no_video_input_720p` 和 `video_input_720p` 即可，不用改代码。只配了 `no_video_input_720p` 而没配 `video_input_720p` 时，带视频输入的 720p 请求回落到 `video_input`

分辨率字符串会先经 `_video_resolution_to_cost_field_suffix` 归一化（转小写、只保留字母数字和下划线、超过 24 字符则丢弃），所以键名统一用小写

### 5.3 完整例子（Seedance 2.0，四个分辨率档）

```json
"volcengine/doubao-seedance-2-0-260128": {
    "litellm_provider": "volcengine",
    "mode": "video_generation",
    "supported_endpoints": ["/v1/videos"],
    "supported_modalities": ["text", "image", "video", "audio"],
    "supported_output_modalities": ["video"],
    "supports_audio_input": true,
    "supports_audio_output": true,
    "video_token_pricing": {
        "no_video_input": 0.000046,
        "no_video_input_1080p": 0.000051,
        "no_video_input_4k": 0.000026,
        "video_input": 0.000028,
        "video_input_1080p": 0.000031,
        "video_input_4k": 0.000016
    },
    "source": "https://www.volcengine.com/docs/82379/1544106",
    "metadata": {"notes": "CN list prices recorded 1:1 as USD"}
}
```

火山官网给的是「元 / 百万 token」，除以 1000000：46 元/百万 -> `0.000046`

### 5.4 只有两档的例子（Fast / Mini）

这两个模型官方只给了单一分辨率价格，就只写基础键，480p 和 720p 都会命中它：

```json
"video_token_pricing": {
    "no_video_input": 0.000037,
    "video_input": 0.000022
}
```

### 5.5 `has_video_input` 从哪来

创建任务时由 `VolcEngineVideoConfig.transform_video_create_request` 判断请求 `content` 里有没有 `type: "video_url"`，结论编码进返回的 video id（`encode_video_id_with_provider`），后续查询任务时再从 video id 解回来。所以查询链路上这个标记是稳定的，不依赖火山响应

### 5.6 兜底链

`video_generation_cost` 的优先级（`litellm/llms/openai/cost_calculation.py:186`）：

1. `video_token_pricing` 命中且 `completion_tokens` 不为 None -> 按 token 算
2. `output_cost_per_video_per_second` -> `单价 × duration_seconds`
3. `output_cost_per_second_{resolution}`（如 `output_cost_per_second_1080p`）-> `单价 × duration_seconds`
4. `output_cost_per_second` -> `单价 × duration_seconds`
5. 都没有 -> 返回 0 并打 warning

注意第 3 条和 `video_token_pricing` 的分辨率档不一样：`output_cost_per_second_720p` 这类键会被 `get_model_info` 过滤掉（只有 `_above_{N}[k]_tokens` 形态的键和显式声明过的字段能透传），所以按秒计费的分辨率档目前只有 `output_cost_per_second_1080p` 能用。`video_token_pricing` 是整个 dict 原样透传的，不受这个限制

火山用的是第 1 条。第 2 到 4 条是给按时长计费的 provider（Sora 等）准备的，火山用不上，但如果将来火山出了按秒计费的模型，这条路是通的，不用改代码

注意 `completion_tokens` 为 0 时费用就是 0。火山任务失败不产出视频时 `completion_tokens` 就是 0，正好对上「仅成功产物计费」

### 5.7 `mode` 的类型定义没跟上

`ModelInfoBase.mode` 的 `Literal`（`litellm/types/utils.py:282`）里没有 `video_generation`，但 JSON 里就是这么写的，UI 的模型新增下拉框里也有这一项（`ui/litellm-dashboard/src/components/add_model/add_model_modes.tsx:9`）。因为 `ModelInfoBase` 是 TypedDict，运行时不校验，所以能跑。改 JSON 时照写 `video_generation` 即可，不用管类型定义

---

## 6. Embedding

最简单，只有 token 单价：

```json
"doubao-embedding-large-text-250515": {
    "litellm_provider": "volcengine",
    "mode": "embedding",
    "max_input_tokens": 4096,
    "max_tokens": 4096,
    "input_cost_per_token": 0.0,
    "output_cost_per_token": 0.0,
    "output_vector_size": 2048,
    "metadata": {"notes": "Volcengine Doubao embedding model - text-250515 version with 2048 dimensions"}
}
```

现存条目的单价全是 `0.0`（未配置真实价格）。新增时如果知道真实价格就填上，别继续留 0

---

## 7. 部署级覆盖（config.yaml / Admin UI / 数据库）

除了改全局价格表，还可以在单个 deployment 上覆盖价格。两个位置语义不同：

### 7.1 `litellm_params:` 下的价格字段

`LiteLLM_Params` 本身是 `extra="allow"`，但 router 只会把 `CustomPricingLiteLLMParams`（`litellm/types/utils.py:3082`）里**枚举过的字段名**从 `litellm_params` 复制进计费用的 model_info（`litellm/router.py:8197`），没枚举的字段就算写进去也不会参与算钱。火山常用的这几个在枚举表里：

```yaml
model_list:
  - model_name: seedance-2.0
    litellm_params:
      model: volcengine/doubao-seedance-2-0-260128
      api_key: os.environ/VOLCENGINE_API_KEY
      video_token_pricing:
        no_video_input: 0.000046
        video_input: 0.000028
        no_video_input_1080p: 0.000051
        video_input_1080p: 0.000031
```

在里面的：`input_cost_per_token`、`output_cost_per_token`、`input_cost_per_image`、`output_cost_per_image`、`output_cost_per_image_above_16384_tokens`、`video_token_pricing`、`output_cost_per_second`、`output_cost_per_second_1080p`、`tiered_pricing`、`input_cost_per_token_above_128k/200k/272k/512k_tokens`

`video_token_pricing` 里的自定义分辨率键（`no_video_input_720p` 等）会原样保留，不会被 Pydantic 吃掉

**不**在枚举表里的（写在 `litellm_params` 下不会生效）：火山实际用到的 `input_cost_per_token_above_32768_tokens` / `_above_131072_tokens` 及其 output、cache 对应键，以及 `output_cost_per_image_above_16384_tokens` 之外的其他图片阈值档。这类字段放到 `model_info:` 下

### 7.2 `model_info:` 下的价格字段

`ModelInfo`（`litellm/types/router.py:136`）是 `extra="allow"`，任意键都能透传，所以 7.1 里那些「不在枚举表里」的字段要放这里：

```yaml
model_list:
  - model_name: seedream-5.0-pro
    litellm_params:
      model: volcengine/doubao-seedream-5-0-pro-260628
      api_key: os.environ/VOLCENGINE_API_KEY
    model_info:
      mode: image_generation
      input_cost_per_image: 0.02
      output_cost_per_image: 0.3
      output_cost_per_image_above_16384_tokens: 0.6
```

`model_info` 里的 `mode` 必须写对：预扣预算的图片分支严格按 `mode in ("image_generation", "image_edit")` 判断（`litellm/proxy/spend_tracking/budget_reservation.py:1089`），写错就回落到 token 路径，预留金额会差一个数量级

---

## 8. 新增或改价的操作清单

1. 到火山官方定价页拿到全部档位价格，记下 URL 作为 `source`
2. 判断落哪种 `mode`，按第 3 到 6 节挑字段
3. 单价换算：官网「元/百万 token」除以 1000000；「元/张」直接写。人民币数字 1:1 记成美元数字
4. 同时改 `model_prices_and_context_window.json` 和 `litellm/model_prices_and_context_window_backup.json`，两边内容必须一致
5. 在 `metadata.notes` 里写清楚原始人民币标价和换算口径
6. 加回归测试。测试放在 `tests/test_litellm/llms/volcengine/<子模块>/`，命名跟随目录里已有文件：
   - chat 分档：断言各档边界（阈值 - 1、阈值、阈值 + 1）算出来的单价正确
   - 图片：断言 `output_tokens = 16384` 和 `16385` 分别落普通档和大图档
   - 视频：断言 `no_video_input` / `video_input` 与 1080p、4k 后缀各自命中
   - 新增视频模型还要在 `test_volcengine_video_transformation.py::test_seedance_model_is_registered_for_video_generation` 的 `SEEDANCE_VIDEO_MODELS` 里加上
7. 跑第 9 节的验证脚本，跟火山官方账单对一遍
8. `make pre-commit`，然后跑测试

---

## 9. 本地验证方法

价格表改完之后，直接在本地算一遍费用，跟火山账单对：

```bash
# chat 分档
.venv/bin/python -c "
from litellm.cost_calculator import cost_per_token
for pt in (10000, 32000, 32001, 200000):
    print(pt, cost_per_token(model='volcengine/<model>', custom_llm_provider='volcengine',
                             prompt_tokens=pt, completion_tokens=1000))
"

# 视频
.venv/bin/python -c "
from litellm.llms.openai.cost_calculation import video_generation_cost
print(video_generation_cost(model='doubao-seedance-2-0-260128', duration_seconds=4.0,
                            custom_llm_provider='volcengine', video_resolution='1080p',
                            completion_tokens=196425, has_video_input=False))
"

# 图片
.venv/bin/python -c "
from litellm.llms.volcengine.image_generation.cost_calculator import cost_calculator
from litellm.types.utils import ImageResponse, ImageObject, ImageUsage, ImageUsageInputTokensDetails
r = ImageResponse(
    data=[ImageObject(url='x')],
    usage=ImageUsage(input_tokens=0,
                     input_tokens_details=ImageUsageInputTokensDetails(image_tokens=0, text_tokens=0),
                     output_tokens=18000, total_tokens=18000),
    hidden_params={'generated_images': 1, 'input_images': 1},
)
print(cost_calculator('doubao-seedream-5-0-pro-260628', r))
"
```

真实链路验证跑本地 proxy：

```bash
python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log
```

然后 curl 打真实火山 API，去 `http://localhost:4000/ui/?page=logs` 看落库的 spend 是否等于手算值。第 5 节的 `usage.completion_tokens` 和第 4 节的 `usage.generated_images` / `usage.output_tokens` 都会出现在响应里，可以直接对账

历史对账数据见同目录的 `volcengine_billing_e2e_report_2026-07-22.md`

---

## 10. 已知限制汇总

| # | 限制 | 影响 | 绕过方式 |
|---|---|---|---|
| 1 | `tiered_pricing` 不参与实际计费 | 只配它等于按 $0 计费 | 必须同时写 `_above_` 字段，见 3.3 |
| 2 | `ModelInfoBase.mode` 的 Literal 缺 `video_generation` | 静态类型和数据对不上，运行时无影响 | 无需处理 |
| 3 | `output_cost_per_second_{resolution}` 只有 `1080p` 能透传 | 按秒计费的其他分辨率档配了不生效 | 火山用不到；真要用得扩 `get_model_info` 的透传规则 |
| 4 | `supported_endpoints` 对这几种模式不做路由拦截 | 填错不会报错，只影响 UI 展示 | 靠人工保证填对 |

已修复：

- 图片阈值不再写死 16384，任意 `output_cost_per_image_above_{N}_tokens` 都能只靠 JSON 配，见 4.4
- 视频分辨率后缀取消白名单，任意分辨率都能单独定价，见 5.2
- `output_cost_per_image_above_16384_tokens` 已加入 `CustomPricingLiteLLMParams`，可在 `litellm_params:` 下覆盖，见 7.1
- 4 个 `doubao-seed-2-0-*` chat 条目从只有 `tiered_pricing`（实际扣费 0）改为 Ark 官方分档的 `_above_` 字段，档位边界纠正为 32768 / 131072
- 补齐 `doubao-seedream-4-0-250828`（0.20/张）和 `doubao-seedream-4-5-251128`（0.25/张），此前完全没有价格条目

## 11. Ark 文档 MCP

价格和档位以方舟文档 MCP 为准，不要凭记忆或旧配置改价：

```json
{
  "mcpServers": {
    "ark-docs-mcp": {
      "url": "https://mcp.ark-doc-resources.cn/mcp/"
    }
  }
}
```

`ark_get_model` 返回的 `pricing[]` 里，`usage_type` 是计费项，`billing_condition` 是档位条件：

| Ark `usage_type` | 含义 | 对应 LiteLLM 字段 |
|---|---|---|
| `InferencePrompt` | 输入 token | `input_cost_per_token` |
| `InferenceCompletion` | 输出 token | `output_cost_per_token` |
| `ContextSessionHit` | 上下文缓存命中 | `cache_read_input_token_cost` |
| `T2ICompletion` / `I2ICompletion` | 文生图 / 图生图，按张 | `output_cost_per_image` |
| `ToICompletion` / `ToILargeCompletion` | 普通输出图 / 大图输出，按张 | `output_cost_per_image` / `output_cost_per_image_above_16384_tokens` |
| `ToIPrompt` | 输入图，按张 | `input_cost_per_image` |
| `NV2VCompletion` / `NV2V1080Completion` / `NV2V4KCompletion` | 无视频输入，按分辨率 | `video_token_pricing.no_video_input[_1080p\|_4k]` |
| `V2VCompletion` / `V2V1080Completion` / `V2V4KCompletion` | 有视频输入，按分辨率 | `video_token_pricing.video_input[_1080p\|_4k]` |

`billing_condition.max_prompt_tokens` 就是 `_above_` 字段该用的阈值。单位 `千tokens` 除以 1000 得每 token 单价，`张` 直接用

`Fast*` / `Batch*` / `Finetune*` / `Audio*` 前缀分别是极速版、批量推理、精调模型和音频模态的价格，当前价格表都没接，需要时再映射到 `_priority` 后缀、`*_batches` 和 `*_audio_token` 系列字段
