# 本地压缩代理

LiteLLM 默认不启用压缩代理。设置 `COMPRESSION_BACKEND=headroom` 或 `COMPRESSION_BACKEND=lean-ctx` 可显式启用对应代理，但 LiteLLM 不会负责启动或停止它们。启用代理时只加载 Headroom guardrail 的压缩和 CCR 召回链路，不加载 `compression_interception`。请先在单独的终端启动所选代理，再运行 `./scripts/start_local.sh`

`start_local.sh` 会先构建最新 Dashboard 源码，再由 LiteLLM 在 `4000` 端口直接提供页面。Token 分析页地址为 `http://localhost:4000/ui/token-analytics/`

## 启动 Headroom

复制下面一行即可在前台启动带 CCR 召回的 token 模式，默认地址是 `http://127.0.0.1:8787`

```bash
cd /path/to/ai-gateway && headroom proxy --mode token
```

不要使用 `--lossless` 测试这条链路；Headroom 的 lossless 模式不会生成 CCR marker，因此 LiteLLM 不会注入 `headroom_retrieve` 或进行二次召回

在另一个终端切换并启动 LiteLLM：

```bash
cd /path/to/ai-gateway && COMPRESSION_BACKEND=headroom ./scripts/start_local.sh
```

也可以把选择写入仓库根目录的 `.env`，之后只需运行 `./scripts/start_local.sh`：

```dotenv
COMPRESSION_BACKEND=headroom
```

## 启动 lean-ctx

复制下面整段即可应用当前的 unwrap + CCR 配置，并在后台启动代理，默认地址是 `http://127.0.0.1:4444`

```bash
cd /path/to/ai-gateway && \
lean-ctx config set crush_verbatim_json true && \
lean-ctx config set compression_level standard && \
lean-ctx config set reference_results true && \
lean-ctx config set archive.enabled true && \
lean-ctx config set archive.ephemeral true && \
lean-ctx config set archive.ephemeral_min_tokens 600 && \
lean-ctx config set archive.threshold_chars 400 && \
lean-ctx proxy start --port=4444 --detach
```

在另一个终端切换并启动 LiteLLM：

```bash
cd /path/to/ai-gateway && COMPRESSION_BACKEND=lean-ctx ./scripts/start_local.sh
```

持久化选择：

```dotenv
COMPRESSION_BACKEND=lean-ctx
```

## 自定义地址和令牌

默认地址可以通过 `.env` 或当前终端环境覆盖：

```dotenv
COMPRESSION_PROXY_URL=http://127.0.0.1:9000
COMPRESSION_PROXY_TOKEN=your-token
```

未设置 `COMPRESSION_PROXY_URL` 时，Headroom 使用端口 `8787`，lean-ctx 使用端口 `4444`。也可以分别用 `HEADROOM_PROXY_PORT` 或 `LEAN_CTX_PROXY_PORT` 改默认端口

## 检查状态

```bash
curl --fail http://127.0.0.1:8787/health
```

```bash
curl --fail --request POST http://127.0.0.1:4444/v1/compress \
  --header "Authorization: Bearer $(lean-ctx proxy token)" \
  --header "Content-Type: application/json" \
  --data '{"messages":[{"role":"user","content":"health check"}],"model":"gpt-4o-mini"}'
```

LiteLLM 配置了 `unreachable_fallback: fail_open`。所选压缩代理未运行时，请求仍会继续，但不会压缩上下文
