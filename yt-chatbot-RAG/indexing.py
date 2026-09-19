'''
Indexing + Streamlit UI:
- fetch transcript from YouTube
- split into chunks, build FAISS embeddings
- run retrieval + LLM chain
- query via Streamlit
'''

from typing import List

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def fetch_transcript(video_id: str, languages: List[str] = ["en"]) -> str:
    """Fetch transcript text for a YouTube video id."""
    yt_api = YouTubeTranscriptApi()
    transcript = yt_api.fetch(video_id=video_id, languages=languages)
    return " ".join(chunk["text"] if isinstance(chunk, dict) else chunk.text for chunk in transcript)


@st.cache_data(show_spinner=False)
def build_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 1) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)


@st.cache_resource(show_spinner=False)
def build_vector_store(chunks: List[str]):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    return FAISS.from_texts(chunks, embeddings)


def create_retriever(vector_store, k: int = 4):
    return vector_store.as_retriever(search_type="similarity", kwargs={"k": k})


def get_context_from_docs(retriever_docs):
    return "\n\n".join(doc.page_content for doc in retriever_docs)


def create_qa_chain(retriever, llm_model="gemini-2.5-flash"):
    llm = ChatGoogleGenerativeAI(model=llm_model)
    prompt = PromptTemplate(
        template=(
            "You are a helpful assistant. Answer only from the given transcript context. "
            "If there is insufficient context, respond 'lack of context'.\n\n" 
            "Transcript:\n{transcript}\n\nQuestion: {question}"
        ),
        input_variables=["transcript", "question"],
    )
    parser = StrOutputParser()

    def query_func(query_text: str):
        retriever_docs = retriever.invoke(query_text)
        context = get_context_from_docs(retriever_docs)

        parallelchain = RunnableParallel(
            {
                "transcript": retriever | RunnableLambda(get_context_from_docs),
                "question": RunnablePassthrough(),
            }
        )

        mainchain = parallelchain | prompt | llm | parser
        return mainchain.invoke(query_text), retriever_docs

    return query_func


def run_pipeline(video_id: str, query: str, k: int = 4):
    fetched = fetch_transcript(video_id)
    chunks = build_chunks(fetched)
    vector_store = build_vector_store(chunks)
    retriever = create_retriever(vector_store, k=k)
    qa_runner = create_qa_chain(retriever)
    answer, docs = qa_runner(query)
    return answer, docs, chunks


def app():
    st.set_page_config(page_title="YT RAG Chatbot", layout="wide")
    st.title("YouTube RAG Chatbot (Streamlit)")

    with st.sidebar:
        st.header("Inputs")
        video_id = st.text_input("YouTube Video ID", "JL9A8ELkNBc")
        question = st.text_input("Question", "What is the solution given in this video to become not replaceable by AI?")
        k = st.slider("Retriever top K", min_value=1, max_value=10, value=4)
        run = st.button("Run QA")

    if run and video_id and question:
        with st.spinner("Fetching transcript, building index, and answering..."):
            try:
                answer, docs, chunks = run_pipeline(video_id, question, k=k)
                st.success("Answer generated")
                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved Contexts")
                for i, doc in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {i}:** {doc.page_content[:700]}{'...' if len(doc.page_content) > 700 else ''}")

                st.subheader("Corpus Stats")
                st.write({"chunks": len(chunks), "retrieved_docs": len(docs)})

            except Exception as e:
                st.error(f"Error: {e}")

        st.info("If the transcript fetch fails, check if the video has captions and the video ID is correct.")


if __name__ == "__main__":
    app()
