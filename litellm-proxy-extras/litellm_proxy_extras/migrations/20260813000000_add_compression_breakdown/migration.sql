ALTER TABLE "LiteLLM_DailyUserSpend"
    ADD COLUMN IF NOT EXISTS "compression_gross_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_gross_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE "LiteLLM_DailyOrganizationSpend"
    ADD COLUMN IF NOT EXISTS "compression_gross_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_gross_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE "LiteLLM_DailyEndUserSpend"
    ADD COLUMN IF NOT EXISTS "compression_gross_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_gross_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE "LiteLLM_DailyAgentSpend"
    ADD COLUMN IF NOT EXISTS "compression_gross_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_gross_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE "LiteLLM_DailyTeamSpend"
    ADD COLUMN IF NOT EXISTS "compression_gross_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_gross_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;

ALTER TABLE "LiteLLM_DailyTagSpend"
    ADD COLUMN IF NOT EXISTS "compression_gross_saved_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_tokens" BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS "compression_gross_savings_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS "compression_extra_input_spend" DOUBLE PRECISION NOT NULL DEFAULT 0.0;
