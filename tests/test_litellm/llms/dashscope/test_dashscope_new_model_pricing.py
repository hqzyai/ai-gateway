import litellm
import pytest

from litellm.types.rerank import RerankResponse
from litellm.types.utils import TranscriptionResponse


@pytest.fixture(autouse=True)
def local_model_cost_map(monkeypatch):
    original = litellm.model_cost
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")
    litellm.get_model_info.cache_clear()
    try:
        yield
    finally:
        litellm.model_cost = original
        litellm.get_model_info.cache_clear()


def test_dashscope_audio_embedding_and_rerank_prices() -> None:
    expected = {
        "dashscope/qwen-audio-3.0-asr-flash": ("input_cost_per_second", 0.00022),
        "dashscope/qwen-audio-3.0-asr-flash-filetrans": ("input_cost_per_second", 0.00022),
        "dashscope/qwen-audio-3.0-asr-flash-streaming": ("input_cost_per_second", 0.00033),
        "dashscope/qwen-audio-3.0-tts-flash": ("input_cost_per_character", 0.0001),
        "dashscope/qwen-audio-3.0-tts-plus": ("input_cost_per_character", 0.00014),
        "dashscope/qwen3.7-text-embedding": ("input_cost_per_token", 0.5e-6),
        "dashscope/text-embedding-v1": ("input_cost_per_token", 0.7e-6),
        "dashscope/qwen3-vl-embedding": ("input_cost_per_image_token", 1.8e-6),
        "dashscope/tongyi-embedding-vision-flash": ("input_cost_per_token", 0.15e-6),
        "dashscope/multimodal-embedding-v1": ("input_cost_per_image_token", 0.9e-6),
        "dashscope/qwen3-rerank": ("input_cost_per_token", 0.5e-6),
        "dashscope/qwen3-vl-rerank": ("input_cost_per_image_token", 1.8e-6),
        "dashscope/gte-rerank-v2": ("input_cost_per_token", 0.8e-6),
    }

    for model, (field, price) in expected.items():
        assert litellm.model_cost[model][field] == price


def test_dashscope_audio_and_multimodal_rerank_costs() -> None:
    transcription = TranscriptionResponse(text="你好")
    transcription._hidden_params["audio_transcription_duration"] = 10.0
    assert litellm.completion_cost(
        completion_response=transcription,
        model="dashscope/qwen-audio-3.0-asr-flash-streaming",
        call_type="transcription",
    ) == pytest.approx(0.0033)
    assert litellm.completion_cost(
        model="dashscope/qwen-audio-3.0-tts-plus",
        prompt="你好",
        call_type="speech",
    ) == pytest.approx(0.00028)

    rerank = RerankResponse(
        id="rerank-request",
        results=[],
        meta={
            "billed_units": {"total_tokens": 100, "input_tokens": 100, "image_tokens": 20},
            "tokens": {"input_tokens": 100, "image_tokens": 20},
        },
    )
    assert litellm.completion_cost(
        completion_response=rerank,
        model="dashscope/qwen3-vl-rerank",
        call_type="rerank",
    ) == pytest.approx(80 * 0.7e-6 + 20 * 1.8e-6)


def test_dashscope_tripo_price_matrix() -> None:
    h31_entry = litellm.model_cost["dashscope/Tripo/Tripo-H3.1"]
    p10_entry = litellm.model_cost["dashscope/Tripo/Tripo-P1.0"]
    h31 = h31_entry["video_token_pricing"]
    p10 = p10_entry["video_token_pricing"]

    assert h31_entry["video_token_pricing_unit"] == "per_generation"
    assert p10_entry["video_token_pricing_unit"] == "per_generation"
    assert litellm.get_model_info("Tripo/Tripo-H3.1", custom_llm_provider="dashscope")[
        "video_token_pricing_unit"
    ] == "per_generation"

    assert h31 == {
        "no_video_input_standard_no_texture": 0.7,
        "video_input_standard_no_texture": 1.4,
        "no_video_input_standard_sd_texture": 1.4,
        "video_input_standard_sd_texture": 2.1,
        "no_video_input_standard_hd_texture": 2.1,
        "video_input_standard_hd_texture": 2.8,
        "no_video_input_ultra_no_texture": 2.1,
        "video_input_ultra_no_texture": 2.8,
        "no_video_input_ultra_sd_texture": 2.8,
        "video_input_ultra_sd_texture": 3.5,
        "no_video_input_ultra_hd_texture": 3.5,
        "video_input_ultra_hd_texture": 4.2,
    }
    assert p10 == {
        "no_video_input_standard_no_texture": 2.1,
        "video_input_standard_no_texture": 2.8,
        "no_video_input_standard_sd_texture": 2.8,
        "video_input_standard_sd_texture": 3.5,
        "no_video_input_standard_hd_texture": 3.5,
        "video_input_standard_hd_texture": 4.2,
    }
