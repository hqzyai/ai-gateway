# SiliconFlow 图片与视频计费实测报告

- 测试日期：2026-07-22
- 测试范围：文生图、图生图、文生视频
- 适配器回归：8 passed
- 结论：3 个真实生成场景均调用成功，厂商账单与 LiteLLM 费用数值 3/3 一致

## 一、结论

当前 SiliconFlow 多模态生成计费按固定单产物价格计算：

```text
图片费用 = 成功生成图片数 × output_cost_per_image
视频费用 = 成功视频任务数 × output_cost_per_video
```

本轮真实生成调用共消耗：

```text
¥0 + ¥0.30 + ¥2.00 = ¥2.30
```

厂商账单以人民币展示；当前 LiteLLM 价格配置按相同数值记录费用。因此，本报告确认的是费用数值一致，不代表系统执行了人民币到美元的汇率换算。

## 二、对账方法

1. 使用 LiteLLM Virtual Key 请求网关，不在客户端直接使用厂商 Key。
2. 由网关使用服务端 SiliconFlow 凭据向厂商发起请求。
3. 确认图片响应包含有效 `data[].url`，或视频状态由 `queued`、`in_progress` 进入 `completed`。
4. 读取 LiteLLM 记录的模型、成功产物数和费用。
5. 在 SiliconFlow 厂商账单中按脱敏 Key 与模型名定位同一轮调用。
6. 比较厂商扣费与 LiteLLM 计算结果。

报告未保存真实 API Key、生成产物 URL 或视频 requestId。

## 三、真实调用结果

| 场景 | 模型 | 结果 | 成功产物数 | 厂商明细 | LiteLLM 费用 | 对账 |
|---|---|---:|---:|---:|---:|---:|
| 文生图 | `Kwai-Kolors/Kolors` | 成功 | 1 张 | ¥0 | 0 | 一致 |
| 图生图 | `Qwen/Qwen-Image-Edit-2509` | 成功 | 1 张 | ¥0.30 | 0.30 | 一致 |
| 文生视频 | `Wan-AI/Wan2.2-T2V-A14B` | `queued → in_progress → completed` | 1 个 | ¥2.00 | 2.00 | 一致 |
| **合计** |  |  | **3 个产物** | **¥2.30** | **2.30** | **3/3 一致** |

同一轮还验证了 Chat `Qwen/Qwen3-8B` 与 Embedding `BAAI/bge-m3` 可用，厂商明细中两项均为 0；它们不计入本报告的图片与视频生成合计。

## 四、场景明细

### 4.1 文生图

- 网关端点：`POST /v1/images/generations`
- 模型：`Kwai-Kolors/Kolors`
- 结果：返回 1 个可用图片 URL
- 当前配置：`output_cost_per_image = 0`
- LiteLLM 计算：`1 × 0 = 0`
- SiliconFlow 明细：¥0
- 结论：一致

图片计数优先使用适配器隐藏参数 `generated_images`；如果该字段缺失，则回退到 `len(response.data)`。

### 4.2 图生图

- 网关端点：`POST /v1/images/edits`
- 模型：`Qwen/Qwen-Image-Edit-2509`
- 结果：上传图片在网关内转为 data URI，以 JSON 调用 SiliconFlow `/images/generations`，返回 1 个可用图片 URL
- 当前配置：`output_cost_per_image = 0.3`
- LiteLLM 计算：`1 × 0.3 = 0.3`
- SiliconFlow 明细：¥0.30
- 结论：一致

该模型支持 1–3 张输入图。普通 `Qwen/Qwen-Image-Edit` 只允许一张输入图。

### 4.3 文生视频

- 网关创建端点：`POST /v1/videos`
- 网关查询端点：`GET /v1/videos/{video_id}`
- 上游创建端点：`POST /video/submit`
- 上游查询端点：`POST /video/status`
- 模型：`Wan-AI/Wan2.2-T2V-A14B`
- 状态：`queued → in_progress → completed`
- 完成时计数：`usage.generated_videos = 1`
- 当前配置：`output_cost_per_video = 2.0`
- LiteLLM 计算：`1 × 2.0 = 2.0`
- SiliconFlow 明细：¥2.00
- 结论：一致

排队、处理中或失败的视频任务不会形成成功视频计数。当前适配器支持完成后通过 `GET /v1/videos/{video_id}/content` 下载视频，但不支持视频列表、删除、remix 或 content variant。

## 五、当前多模态价格配置

价格来自仓库根目录 `model_prices_and_context_window.json` 的 SiliconFlow 模型条目。

### 图片

| 模型 | 配置字段 | 当前值 |
|---|---|---:|
| `Kwai-Kolors/Kolors` | `output_cost_per_image` | 0 |
| `Qwen/Qwen-Image` | `output_cost_per_image` | 0.3 |
| `Qwen/Qwen-Image-Edit` | `output_cost_per_image` | 0.3 |
| `Qwen/Qwen-Image-Edit-2509` | `output_cost_per_image` | 0.3 |
| `Tongyi-MAI/Z-Image-Turbo` | `output_cost_per_image` | 0.1 |
| `Tongyi-MAI/Z-Image` | `output_cost_per_image` | 0.3 |
| `baidu/ERNIE-Image-Turbo` | `output_cost_per_image` | 0.11 |

### 视频

| 模型 | 配置字段 | 当前值 |
|---|---|---:|
| `Wan-AI/Wan2.2-T2V-A14B` | `output_cost_per_video` | 2.0 |
| `Wan-AI/Wan2.2-I2V-A14B` | `output_cost_per_video` | 2.0 |

## 六、自动化回归

本轮执行：

```bash
uv run pytest -q \
  tests/test_litellm/llms/siliconflow/image_generation \
  tests/test_litellm/llms/siliconflow/image_edit \
  tests/test_litellm/llms/siliconflow/videos
```

结果：

```text
........                                                                 [100%]
8 passed in 0.19s
```

这些测试覆盖图片请求参数映射、图片编辑输入约束、响应标准化、视频创建、状态映射、内容下载和不支持能力。自动化测试不替代厂商真实账单检查，因此本报告同时保留了三项真实调用对账结果。

## 七、运维建议

- 厂商 Key 只保存在服务端环境变量或密钥管理系统中，客户端始终使用 LiteLLM Virtual Key。
- 价格变更时，同时记录价格表版本、生效时间和来源，避免历史账单无法复算。
- 修改价格后，用一个低成本图片模型和一个视频模型各做一次小额验证。
- 若改为远程价格表或启动时同步，应增加校验和、缓存与回滚策略，并确认运行中进程何时刷新缓存。
- 对 0 单价模型也保留成功产物计数；费用为 0 不代表请求未执行。
