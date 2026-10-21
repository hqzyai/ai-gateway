export const defaultPricingProviders: readonly string[] = ["siliconflow", "volcengine", "dashscope"];

export const modelProviders: readonly string[] = [...defaultPricingProviders, "openai"];

export const isDefaultPricingProvider = (provider: string): boolean => defaultPricingProviders.includes(provider);

export const isModelProvider = (provider: string): boolean => modelProviders.includes(provider);
