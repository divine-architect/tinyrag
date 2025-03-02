import numpy as np
import ollama
from typing import Optional, Any
from sklearn.metrics.pairwise import cosine_similarity

class TinyRAG_Ollama:
    def __init__(self, embedding_model: str = "nomic-embed-text", llm_model: Optional[str] = None) -> None:
        self.embedding_model: str = embedding_model
        self.llm_model: Optional[str] = llm_model
        self.documents: list[str] = []
        self.embeddings: list[Optional[list[float]]] = []
        self.prev_responses: list[str] = []
        
    def add_documents(self, documents: list[str]) -> None:
        if not documents:
            return
            
        self.documents.extend(documents)
        
        for doc in documents:
            try:
                response: dict[str, Any] = ollama.embeddings(model=self.embedding_model, prompt=doc)
                self.embeddings.append(response['embedding'])
            except Exception as e:
                print(f"Error embedding document: {str(e)}")
                self.embeddings.append(None)
    
    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        if not self.documents:
            return []
            
        try:
            query_embedding_resp: dict[str, Any] = ollama.embeddings(model=self.embedding_model, prompt=query)
            query_embedding: list[float] = query_embedding_resp['embedding']
        except Exception as e:
            print(f"Error embedding query: {str(e)}")
            return []
            
        valid_docs: list[str] = []
        valid_embeddings: list[list[float]] = []
        for i, emb in enumerate(self.embeddings):
            if emb is not None:
                valid_docs.append(self.documents[i])
                valid_embeddings.append(emb)
        
        if not valid_embeddings:
            return []
            
        query_embedding_np: np.ndarray = np.array(query_embedding).reshape(1, -1)
        docs_embeddings_np: np.ndarray = np.array(valid_embeddings)
        
        try:
            similarities: np.ndarray = cosine_similarity(query_embedding_np, docs_embeddings_np)[0]
            top_k = min(top_k, len(valid_docs))
            top_indices: np.ndarray = np.argsort(similarities)[::-1][:top_k]
            
            return [(valid_docs[i], similarities[i]) for i in top_indices]
        except Exception as e:
            print(f"Error calculating similarities: {str(e)}")
            return []
    
    def query(self, query: str, top_k: int = 3, temperature: float = 0.7, max_tokens:int = 1000 ) -> str:
        try:
            relevant_docs: list[tuple[str, float]] = self.retrieve(query, top_k)
            
            if not relevant_docs:
                return "No relevant information found."
            
            related_info_from_docs: str = "\n\n".join([f"Doc (score: {score:.2f}):\n{doc}" for doc, score in relevant_docs])
            
            conversation_history: str = "\n\n".join(self.prev_responses) if self.prev_responses else "No history available."
            
            prompt: str = f"""Answer based on this info only:
            Info from documents:
            {related_info_from_docs}
            Conversation History:
            {conversation_history}
            Question: {query}

            Answer:"""
            
            
            response: dict[str, Any] = ollama.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "max_tokens": max_tokens}
            )
            self.prev_responses.append(f"Question: {query}\n\n Answer: {response['message']['content']}")
            
            if len(self.prev_responses) > 5:
                self.prev_responses.pop(0)
                
            return response['message']['content']
        except Exception as e:
            print(f"Error during query: {str(e)}")
            return "An error occurred while processing your query."
