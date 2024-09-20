import torch
from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelBackendConfig, ModelChatConfig, ModelConfig, ModelFrontendConfig

default_chat_config = ModelChatConfig(
    max_session_length=8192,
    sep_token="###",
    stop_token="###",
    extra_stop_sequences=["</s>"],
    generation_params=dict(do_sample=1, temperature=0.6, top_p=0.9),
)

MODEL_FAMILIES = {
    "BLOOM": [
        ModelConfig(
            ModelBackendConfig(repository="bigscience/bloom-560m"),
            ModelFrontendConfig(
                name="BLOOM-560B",
                model_card="https://huggingface.co/bigscience/bloom-560m",
                license="https://bit.ly/bloom-license",
            ),
            ModelChatConfig(
                max_session_length=2048,
                sep_token="\n\n",
                stop_token="</s>",
                extra_stop_sequences=["\nHuman"],
                generation_params=default_chat_config.generation_params,
            ),
        ),
    ],
    "Llama 3": [
        ModelConfig(
            ModelBackendConfig(repository="meta-llama/Meta-Llama-3.1-8B-Instruct"),
            ModelFrontendConfig(
                name="Llama 3.1 8B IT",
                model_card="https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct",
                license="https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE",
            ),
            ModelChatConfig(
                max_session_length=2048,
                sep_token="\n\n",
                stop_token=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                extra_stop_sequences=["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                generation_params=default_chat_config.generation_params,
            ),
        ),
        # ModelConfig(
        #     ModelBackendConfig(repository="meta-llama/Llama-2-70b-chat-hf"),
        #     ModelFrontendConfig(
        #         name="Llama 2 (70B-Chat)",
        #         model_card="https://huggingface.co/meta-llama/Llama-2-70b-chat-hf",
        #         license="https://bit.ly/llama2-license",
        #     ),
        #     default_chat_config,
        # ),
    ],
    # "Falcon": [
    #     ModelConfig(
    #         ModelBackendConfig(repository="tiiuae/falcon-180B-chat", public_api=False),
    #         ModelFrontendConfig(
    #             name="Falcon 180B-Chat",
    #             model_card="https://huggingface.co/tiiuae/falcon-180B-chat",
    #             license="https://huggingface.co/spaces/tiiuae/falcon-180b-license/blob/main/LICENSE.txt",
    #         ),
    #         ModelChatConfig(
    #             max_session_length=8192,
    #             sep_token="\n",
    #             stop_token="\n",
    #             extra_stop_sequences=["<|endoftext|>", "\nFalcon:", " Falcon:", "\nUser:", " User:", "###"],
    #             generation_params=dict(do_sample=1, temperature=0.75, top_p=0.9, repetition_penalty=1.2),
    #         ),
    #     ),
    # ],
    # "Llama": [
    #     ModelConfig(
    #         ModelBackendConfig(repository="huggyllama/llama-65b", adapter="timdettmers/guanaco-65b"),
    #         ModelFrontendConfig(
    #             name="Guanaco-65B",
    #             model_card="https://huggingface.co/timdettmers/guanaco-65b",
    #             license="https://huggingface.co/timdettmers/guanaco-65b",
    #         ),
    #         default_chat_config,
    #     ),
    #     ModelConfig(
    #         ModelBackendConfig(repository="huggyllama/llama-65b"),
    #         ModelFrontendConfig(
    #             name="Llama-65B",
    #             model_card="https://github.com/facebookresearch/llama/blob/llama_v1/MODEL_CARD.md",
    #             license="https://bit.ly/llama-license",
    #         ),
    #         default_chat_config,
    #     ),
    # ],
    # "BLOOM": [
    #     ModelConfig(
    #         ModelBackendConfig(repository="bigscience/bloomz"),
    #         ModelFrontendConfig(
    #             name="BLOOMZ-176B",
    #             model_card="https://huggingface.co/bigscience/bloomz",
    #             license="https://bit.ly/bloom-license",
    #         ),
    #         ModelChatConfig(
    #             max_session_length=2048,
    #             sep_token="\n\n",
    #             stop_token="</s>",
    #             extra_stop_sequences=["\n\nHuman"],
    #             generation_params=default_chat_config.generation_params,
    #         ),
    #     ),
    # ],
}

INITIAL_PEERS = PUBLIC_INITIAL_PEERS
# Set this to a list of multiaddrs to connect to a private swarm instead of the public one, for example:
INITIAL_PEERS = ['/ip4/172.27.40.250/tcp/31337/p2p/QmTQzBrsF37833RgHz4qUZY3RGk9F2d3y5TKAHDX9LB6mM']

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from cpufeature import CPUFeature

    has_avx512 = CPUFeature["AVX512f"] and CPUFeature["OS_AVX512"]
except ImportError:
    has_avx512 = False

if DEVICE == "cuda":
    TORCH_DTYPE = "auto"
elif has_avx512:
    TORCH_DTYPE = torch.bfloat16
else:
    TORCH_DTYPE = torch.float32  # You can use bfloat16 in this case too, but it will be slow

STEP_TIMEOUT = 5 * 60
