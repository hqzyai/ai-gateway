# Docker Development Guide

This guide provides instructions for building and running the LiteLLM application using Docker and Docker Compose.

## Prerequisites

- Docker
- Docker Compose

## Building and Running the Application

To build and run the application, you will use the `docker-compose.yml` file located in the root of the project. This file is configured to use the `Dockerfile.non_root` for a secure, non-root container environment.

### 1. Set the Master Key

The application requires a `LITELLM_MASTER_KEY` for signing and validating tokens. You must set this key as an environment variable before running the application.

Create a `.env` file in the root of the project and add the following line:

```
LITELLM_MASTER_KEY=your-secret-key
```

Replace `your-secret-key` with a strong, randomly generated secret.

“Add Model”的供应商下拉只提供硅基流动（SiliconFlow）、火山引擎（Volcengine）、DashScope 和 OpenAI。用户通过页面或自己的 `model_list` 配置添加模型，启动时不会自动创建示例模型。使用环境变量配置凭证时，可填写 `SILICONFLOW_API_KEY`、`VOLCENGINE_API_KEY`、`DASHSCOPE_API_KEY` 或 `OPENAI_API_KEY`；`OPENAI_API_BASE` 可设为 OpenAI 官方地址或兼容接口地址

计费规则默认读取镜像内的价格表，只加载硅基流动、火山引擎和 DashScope 已有的规则；刷新、定时刷新和远程加载失败后的本地回退也保持这个范围。`LITELLM_LOCAL_MODEL_COST_MAP=False` 可显式开启远程价格数据源，供应商范围仍保持不变。页面的计费列表、供应商筛选、模式筛选和统计卡片均只计算这三家及用户手动添加的 OpenAI 规则

OpenAI 不预设价格，包括零价格；用户需要在“计费规则”中按实际模型标识（`openai/模型名`）手动添加规则，保存后随数据库配置持久化，刷新价格表时继续保留

已有数据库中的模型和自定义计费规则不会被删除；不在以上供应商范围内的规则不在此页面展示。重新构建并部署前端后，下拉框和计费列表的限制生效

### 2. Build and Run the Containers

Once you have set the `LITELLM_MASTER_KEY`, you can build and run the containers using the following command:

```bash
docker compose up -d --build
```

This command will:

-   Build the Docker image using `Dockerfile.non_root`.
-   Start the `litellm`, `litellm_db`, and `prometheus` services in detached mode (`-d`).
-   The `--build` flag ensures that the image is rebuilt if there are any changes to the Dockerfile or the application code.

### 3. Verifying the Application is Running

You can check the status of the running containers with the following command:

```bash
docker compose ps
```

To view the logs of the `litellm` container, run:

```bash
docker compose logs -f litellm
```

### 4. Stopping the Application

To stop the running containers, use the following command:

```bash
docker compose down
```

## Hardened / Offline Testing

To ensure changes are safe for non-root, read-only root filesystems and restricted egress, always validate with the hardened compose file:

```bash
docker compose -f docker-compose.yml -f docker-compose.hardened.yml build --no-cache
docker compose -f docker-compose.yml -f docker-compose.hardened.yml up -d
```

This setup:
- Builds from `docker/Dockerfile.non_root` with Prisma engines and Node toolchain baked into the image.
- Runs the proxy as a non-root user with a read-only rootfs and only writable tmpfs mounts:
  - `/app/cache` (Prisma/NPM cache; backing `PRISMA_BINARY_CACHE_DIR`, `NPM_CONFIG_CACHE`, `XDG_CACHE_HOME`)
  - `/app/migrations` (Prisma migration workspace; backing `LITELLM_MIGRATION_DIR`)
- Pre-builds and serves the admin UI from read-only paths:
  - `/var/lib/litellm/ui` (pre-restructured Next.js UI with `.litellm_ui_ready` marker)
  - `/var/lib/litellm/assets` (UI logos and assets)
- Routes all outbound traffic through a local Squid proxy that denies egress, so Prisma migrations must use the cached CLI and engines.

You should also verify offline Prisma behaviour with:

```bash
docker run --rm --network none --entrypoint prisma ghcr.io/berriai/litellm:main-stable --version
```

This command should succeed (showing engine versions) even with `--network none`, confirming that Prisma binaries are available without network access.

## Troubleshooting

-   **`build_admin_ui.sh: not found`**: This error can occur if the Docker build context is not set correctly. Ensure that you are running the `docker-compose` command from the root of the project.
-   **`Master key is not initialized`**: This error means the `LITELLM_MASTER_KEY` environment variable is not set. Make sure you have created a `.env` file in the project root with the `LITELLM_MASTER_KEY` defined.
