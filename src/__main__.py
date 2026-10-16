from llm_sdk import Small_LLM_Model
from parser import load_function_definitions, load_prompts


def main() -> None:
    # model name is `Qwen/Qwen3-0.6B` by default
    model = Small_LLM_Model()
    vocab_path = model.get_path_to_vocab_file()
    prompts = load_prompts("data/input/function_calling_tests.json")


if __name__ == "__main__":
    main()