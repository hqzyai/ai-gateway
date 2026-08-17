# Headroom 部署模板

该目录把官方 `headroom-ai` Python 包封装为一个固定版本的容器，并在前面增加只允许 Bearer Token 访问的轻量 sidecar。它不会修改 Headroom 源码

sidecar 对网关提供两个内部接口：

- `POST /v1/compress`
- `GET /v1/retrieve/{hash}`

## 启动

先生成随机 Token，并将示例环境文件复制到仓库外的安全位置：

```bash
openssl rand -hex 32
docker compose --env-file /secure/path/headroom.env -f docker/headroom/compose.yaml up -d --build
```

环境文件格式见 `env.example`。不要把包含真实 Token 的环境文件提交到 Git

## 接入 LiteLLM

LiteLLM 容器和 Headroom 容器应位于同一个 Docker 网络。LiteLLM 需要以下环境变量：

```dotenv
COMPRESSION_PROXY_URL=http://headroom:8788
COMPRESSION_PROXY_TOKEN=与_HEADROOM_PROXY_TOKEN_相同的值
```

Guardrail 配置参考仓库中的 `litellm/proxy/dev_config_headroom.yaml`。启用 Headroom 时不要同时启用 `compression_interception`

`8788` 只应在容器内部网络开放，不应直接映射到公网。Headroom 的状态数据保存在 `headroom-data` volume 中，重新创建容器时应保留该 volume，避免已经生成的 CCR hash 无法召回

## 验证

容器健康后可以运行：

```bash
LITELLM_BASE_URL=http://127.0.0.1:4000 \
LITELLM_MASTER_KEY='<your-master-key>' \
LITELLM_MODEL='<your-model>' \
python docker/headroom/smoke_test.py
```

脚本只从环境变量读取凭据，不会打印密钥或完整模型响应
