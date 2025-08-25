import chromadb
from chromadb.config import Settings
from openai import OpenAI
import tiktoken
import numpy as np
from google import genai

class FinancialSituationMemory:
    def __init__(self, name, config):
        self.embedding = "text-embedding-3-small"
        self.model = "models/gemini-embedding-exp-03-07"
        self.client = genai.Client()
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        self.situation_collection = self.chroma_client.create_collection(name=name)
        # OpenAI's text-embedding-3-small has a limit of 8191 tokens
        self.max_tokens = 8191
        try:
            self.encoding = tiktoken.encoding_for_model(self.embedding)
        except KeyError:
            self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_gemini_tokens(self, text):
        response = self.client.models.count_tokens(model=self.model, contents=text)
        tokens = response.total_tokens
        if tokens is None:
            raise ValueError('no token length returned')
        return tokens
    
    def gemini_embedding(self, text: str):
        pass

    def get_embedding(self, text: str):
        """
        Gemini 임베딩을 가져옵니다.
        텍스트가 모델의 최대 토큰 제한보다 길 경우, 의미 단위로 분할하고 임베딩을 평균냅니다.
        """
        # 먼저 전체 텍스트의 토큰 수를 확인합니다.
        total_tokens = self.count_gemini_tokens(text)

        if total_tokens <= self.max_tokens:
            response = self.client.models.embed_content(model=self.model, contents=text)
            return response.embeddings

        # 텍스트가 너무 길 경우, 의미 단위(문단)로 분할합니다.
        print("텍스트가 너무 깁니다. 의미 단위로 분할하여 처리합니다...")
        chunk_embeddings = []
        
        # 문단 -> 줄바꿈 순서로 텍스트를 나눕니다.
        text_units = text.split('\n\n')
        if len(text_units) <= 1:
            text_units = text.split('\n')

        current_chunk = ""
        for unit in text_units:
            # 빈 유닛은 건너뜁니다.
            if not unit.strip():
                continue

            # 현재 청크에 새 유닛을 더했을 때 토큰 수를 확인합니다.
            temp_chunk = f"{current_chunk}\n{unit}".strip()
            temp_tokens = self.count_gemini_tokens(temp_chunk)

            if temp_tokens <= self.max_tokens:
                # 제한을 넘지 않으면 현재 청크에 추가합니다.
                current_chunk = temp_chunk
            else:
                # 제한을 넘으면, 지금까지의 청크를 임베딩하고 새 청크를 시작합니다.
                response = self.client.models.embed_content(model=self.model, contents=current_chunk)
                chunk_embeddings.append(response.embeddings)
                
                # 새 청크는 현재 유닛으로 시작합니다.
                current_chunk = unit

        # 마지막으로 남은 청크를 처리합니다.
        if current_chunk:
            response = self.client.models.embed_content(model=self.model, contents=current_chunk)
            chunk_embeddings.append(response.embeddings[0])

        if not chunk_embeddings:
            return None

        avg_embedding = np.mean(chunk_embeddings, axis=0)
        return avg_embedding.tolist()
    #def get_embedding(self, text):
    #    """
    #    Get OpenAI embedding for a text.
    #    If the text is longer than the model's maximum token limit, it's chunked and the embeddings are averaged.
    #    """
    #    tokens = self.encoding.encode(text)

    #    if len(tokens) <= self.max_tokens:
    #        response = self.client.embeddings.create(
    #            model=self.embedding, input=text
    #        )
    #        return response.data[0].embedding

    #    # Text is too long, chunk it
    #    chunk_embeddings = []
    #    for i in range(0, len(tokens), self.max_tokens):
    #        chunk_tokens = tokens[i : i + self.max_tokens]
    #        chunk_text = self.encoding.decode(chunk_tokens)

    #        response = self.client.embeddings.create(
    #            model=self.embedding, input=chunk_text
    #        )
    #        chunk_embeddings.append(response.data[0].embedding)

    #    # Average the embeddings of the chunks
    #    avg_embedding = np.mean(chunk_embeddings, axis=0)
    #    return avg_embedding.tolist()

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice. Parameter is a list of tuples (situation, rec)"""

        situations = []
        advice = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            situations.append(situation)
            advice.append(recommendation)
            ids.append(str(offset + i))
            embeddings.append(self.get_embedding(situation))

        self.situation_collection.add(
            documents=situations,
            metadatas=[{"recommendation": rec} for rec in advice],
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results
