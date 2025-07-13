from llm_builder import LLMBuilder

# 1. OpenAI
# llm = LLMBuilder().with_openai(model_name="gpt-4", temperature=0.2).build()

# 2. Azure OpenAI
# llm = LLMBuilder().with_azure_openai().build()

# 3. Claude
# llm = LLMBuilder().with_claude().build()

# 4. Gemini
# llm = LLMBuilder().with_gemini().build()

if __name__ == "__main__":
    llm = LLMBuilder().with_openai(model_name="gpt-4", temperature=0.2).build()
    print(llm.invoke("how is bangalore?"))
