import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(model: str, image_response: ImageResponse) -> float:
    model_info = litellm.get_model_info(model=model, custom_llm_provider=litellm.LlmProviders.SILICONFLOW.value)
    output_cost_per_image = float(model_info.get("output_cost_per_image") or 0.0)
    generated_images = image_response.get_hidden_param("generated_images")
    output_image_count = (
        generated_images
        if isinstance(generated_images, int) and generated_images >= 0
        else len(image_response.data or [])
    )
    return output_cost_per_image * output_image_count
