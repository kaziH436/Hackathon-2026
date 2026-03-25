"""
qa_chain.py
LangChain Q&A over city datasets (CSV + GeoJSON).
Uses LCEL (LangChain Expression Language) — compatible with langchain v1.x
No RetrievalQA, no langchain.chains needed.
"""

import os
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from config import settings


# ── 1. BERT Embeddings ────────────────────────────────────────────────────
def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": settings.DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )


# ── 2. FAISS Vector Store ─────────────────────────────────────────────────
def load_or_build_vectorstore(embeddings: HuggingFaceEmbeddings) -> FAISS:
    index_path = settings.FAISS_INDEX_PATH

    if os.path.exists(index_path):
        print(f"[QAChain] Loading FAISS index from {index_path}")
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    raise FileNotFoundError(
        f"FAISS index not found at '{index_path}'.\n"
        "Run this first:  cd Hackathon-2026/nlp && python data_loader.py"
    )


# ── 3. Generator LLM ─────────────────────────────────────────────────────
def load_generator_llm() -> HuggingFacePipeline:
    tokenizer = AutoTokenizer.from_pretrained(settings.GENERATOR_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        settings.GENERATOR_MODEL,
        torch_dtype=torch.float16 if settings.DEVICE == "cuda" else torch.float32,
    ).to(settings.DEVICE)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        device=0 if settings.DEVICE == "cuda" else -1,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ── 4. Prompt ─────────────────────────────────────────────────────────────
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a city policy analyst assistant.
Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough data."

Context:
{context}

Question: {question}

Answer:""",
)


# ── 5. Format retrieved docs into a single string ─────────────────────────
def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


# ── 6. QA Chain (LCEL) ───────────────────────────────────────────────────
class CityQAChain:
    """
    Uses LCEL pipeline instead of RetrievalQA.
    Compatible with langchain v1.x
    """

    def __init__(self):
        self.embeddings  = load_embeddings()
        self.vectorstore = load_or_build_vectorstore(self.embeddings)
        self.llm         = load_generator_llm()
        self.retriever   = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K_DOCS},
        )

        # LCEL chain: retrieve → format → prompt → llm → parse
        self.chain = (
            {
                "context":  self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | QA_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> dict:
        # Get answer from LCEL chain
        answer = self.chain.invoke(question)

        # Get source docs separately for metadata
        source_docs = self.retriever.invoke(question)
        sources = [
            {
                "source":  doc.metadata.get("source", "unknown"),
                "snippet": doc.page_content[:200],
            }
            for doc in source_docs
        ]

        return {
            "answer":   answer.strip(),
            "sources":  sources,
            "raw_docs": [doc.page_content for doc in source_docs],
        }


# ── Singleton ─────────────────────────────────────────────────────────────
_qa_chain_instance: CityQAChain | None = None


def get_qa_chain() -> CityQAChain:
    global _qa_chain_instance
    if _qa_chain_instance is None:
        _qa_chain_instance = CityQAChain()
    return _qa_chain_instance
