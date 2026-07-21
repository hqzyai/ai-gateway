import litellm
from litellm.types.utils import ImageResponse


def cost_calculator(model: str, image_response: ImageResponse) -> float:
    model_info = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.VOLCENGINE.value,
    )
    input_cost_per_image = float(model_info.get("input_cost_per_image") or 0.0)
    output_cost_per_image = float(model_info.get("output_cost_per_image") or 0.0)
    input_images = image_response.get_hidden_param("input_images")
    generated_images = image_response.get_hidden_param("generated_images")
    input_image_count = input_images if isinstance(input_images, int) and input_images >= 0 else 0
    output_image_count = (
        generated_images
        if isinstance(generated_images, int) and generated_images >= 0
        else len(image_response.data or [])
    )
    return input_cost_per_image * input_image_count + output_cost_per_image * output_image_count
