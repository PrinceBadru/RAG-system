"""
Handles LLM inference via Hugging Face Serverless API (Cloud).
Replaces the heavy local PyTorch pipeline with a lightweight API client.
"""
import os
from huggingface_hub import InferenceClient
from pdf_reader import PdfReader
from local_embedding import LocalEmbedding

class AiModel:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        '''
        Initializes the InferenceClient to talk to Hugging Face's hosted models.
        '''
        self.model_name = model_name
        self.hf_token = os.environ.get("HF_TOKEN")
        
        if not self.hf_token:
            print("WARNING: HF_TOKEN not found in environment variables. API calls may fail or be heavily rate-limited.")
            
        print(f"Connecting to Hugging Face Inference API for model: {self.model_name}")
        # Initialize the lightweight client instead of downloading huge weights
        self.client = InferenceClient(model=self.model_name, token=self.hf_token)

    def ask_a_question(self, prompt="Hello there!"):
        '''
        Basic question asking without RAG.
        '''
        messages = [{"role": "user", "content": prompt}]
        response = self.client.chat_completion(messages=messages, max_tokens=1000)
        print(response.choices[0].message.content)

    def ask_a_question_from_pdf(self, pdf_path, prompt="tell me what is this pdf about"):
        '''
        RAG operation without streaming (waits for full response).
        '''
        pdf_reader = PdfReader(pdf_path)
        pdf_paragraphs = pdf_reader.get_paragraphs()
        
        local_embedding = LocalEmbedding()
        local_embedding.build_index(pdf_paragraphs)

        relevant_sections = local_embedding.get_context(prompt, 10)
        messages = self.build_messages(relevant_sections, prompt)

        response = self.client.chat_completion(messages=messages, max_tokens=1000)
        print(response.choices[0].message.content)

    def ask_a_question_from_pdf_stream(self, pdf_path: str, prompt: str = "tell me what is this pdf about", local_embedding=None):
        '''
        Streaming RAG operation. Yields chunks for Streamlit to render.
        '''
        if local_embedding is None:
            pdf_reader = PdfReader(pdf_path)
            pdf_paragraphs = pdf_reader.get_paragraphs()
            local_embedding = LocalEmbedding()
            local_embedding.build_index(pdf_paragraphs)

        # ARCHITECTURE: VI[(Vector Index In-Memory)] -->|Top K Context| LC[LLM Client]
        relevant_sections = local_embedding.get_context(prompt, k=10)
        messages = self.build_messages(relevant_sections, prompt)

        # ARCHITECTURE: LC[LLM Client] -->|Prompt| HF((Hugging Face Serverless API))
        stream = self.client.chat_completion(
            messages=messages, 
            max_tokens=1000, 
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content is not None:
                yield content

    def build_messages(self, relevant_sections, question_prompt):
        '''
        Constructs the system and user messages for the Chat API.
        '''
        return [
            {
                "role": "system",
                "content": (
                    "You are an AI assistant. Answer the following question based *only* on the provided document text. "
                    "If the answer is not found in the document, say 'The document does not contain information on this topic.' "
                    "Do not use any prior knowledge.\n\n"
                    "Document Text:\n"
                    "---\n"
                    f"{relevant_sections}\n"
                    "---"
                )
            },
            {
                "role": "user",
                "content": f"Question: {question_prompt}"
            }
        ]
