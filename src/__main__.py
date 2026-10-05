from llm_sdk import Small_LLM_Model

def main() -> None:
    model = Small_LLM_Model()
    encoded = model.encode("Hello world!")
    token_ids: list[int] = encoded.tolist()[0]

    print("token IDs:", Token_ids)
    print("Decoded:", model.decode(token_ids))

if __name__ == "__main__":
    main()