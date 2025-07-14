#All necessary imports
#Please read README file on how to use this website/app
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

#Loading the environment variables
load_dotenv(override=True)

class LegalDocAI:
    #Establishing all necessary requirements for OpenAi connection
    #Creating Vector store
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        self.llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0.3,
        )
        self.vectorstore = None
        self.qa_chain = None

    #Function to load and process documents
    #Can process pdf, txt as well as docx (word files)
    def load_and_process_documents(self, folder_path):
        documents = []
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            try:
                if file_name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                elif file_name.endswith(".docx"):
                    loader = Docx2txtLoader(file_path)
                elif file_name.endswith(".txt"):
                    loader = TextLoader(file_path)
                else:
                    continue
                documents.extend(loader.load())
            except Exception as e:
                st.error(f"Error loading {file_name}: {str(e)}")
    #Propogates error message if not found
        if not documents:
            st.error("No valid documents found. Supported formats: PDF, DOCX, TXT.")
            return False

    #Logic for splitting text into chunks
    #Running FAISS algorithm for vector similarity matching
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        self.vectorstore = FAISS.from_documents(
            documents=splits,
            embedding=self.embeddings
        )

    #Creating a template for the type of analysis required
        prompt_template = """
        As a legal AI assistant, analyze the legal document that has been chosen by the User,
        and answer the question that the user asks
        using only the information in the document that has been chosen

        Furthermore, give a  **Risk Assessment** for the chosen legal document and (Rate 1-5⭐ + reason)

        Context: {context}
        Question: {question}
        Structured Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

    #Initalizing the QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        return True

    #Calling the question asking argument flow
    def ask_question(self, question):
        result = self.qa_chain.invoke({"query": question})
        return {
            "answer": result["result"],
            "sources": list(set([doc.metadata["source"] for doc in result["source_documents"]])),
            "context": "\n".join([doc.page_content for doc in result["source_documents"]])
        }

    # Generating solution strategies for risky parts
    def generate_solutions(self, context):
        solution_prompt = PromptTemplate(
            template="""
            Given the following legal document content, identify the riskiest parts (e.g., vague liability, harsh penalties, vague termination).
            Then, provide 2-3 clear solutions or suggestions to mitigate each risky clause (e.g., clause rewrite, add clarification, legal negotiation tips).

            Legal Document Content:
            {context}

            Output Format:
            - **Risky Clause Summary**: ...
              **Suggested Solution(s)**: ...
            """,
            input_variables=["context"]
        )
        chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(),
            chain_type_kwargs={"prompt": solution_prompt}
        )
        return chain.invoke({"query": "Provide risk mitigation strategies", "context": context})["result"]

#Frontend UI generation using StreamLit
def main():
    #Basic Webpage Setup
    st.set_page_config(page_title="LegalDoc AI", page_icon="⚖️")
    st.title("⚖️ LegalDoc AI")
    st.markdown("""
    <style>
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa500; }
    .risk-low { color: #2ecc71; }
    </style>
    """, unsafe_allow_html=True)

    if "legal_ai" not in st.session_state:
        st.session_state.legal_ai = None
        st.session_state.docs_loaded = False
        st.session_state.analysis_response = None

    #Side bar for document loading
    with st.sidebar:
        st.header("📂 Document Setup")
        docs_folder = st.text_input("Path to documents folder", "./legal_docs")

    #Dropdown for some sample questions.
        st.header("💡 Sample Questions")
        sample_questions = [
            "What are the confidentiality obligations?",
            "Highlight termination clauses",
            "What is the liability cap?",
            "Are there any automatic renewals?"
        ]
        selected_query = st.selectbox("Try:", sample_questions)

    #Button to load documents from currently selected path
        if st.button("Load Documents"):
            if not os.path.exists(docs_folder):
                st.error("Folder not found!")
            else:
                with st.spinner("Processing documents..."):
                    legal_ai = LegalDocAI()
                    if legal_ai.load_and_process_documents(docs_folder):
                        st.session_state.legal_ai = legal_ai
                        st.session_state.docs_loaded = True
                        st.success(f"Loaded {len(os.listdir(docs_folder))} documents!")
                    else:
                        st.error("Failed to process documents.")

    if st.session_state.docs_loaded:
        st.subheader("🔍 Ask a Question")
        question = st.text_area("Your question:", value=selected_query, height=100)

    #Button to start the analyzing algorithm 
        if st.button("Analyze", type="primary"):
            with st.spinner("Analyzing contracts..."):
                response = st.session_state.legal_ai.ask_question(question)
                st.session_state.analysis_response = response
                st.subheader("📝 Analysis")
                st.markdown(response["answer"], unsafe_allow_html=True)

                st.subheader("📂 Source Documents")
                for source in response["sources"]:
                    st.write(f"- `{source}`")

        # Button to generate solutions (only available after analysis)
        if st.session_state.analysis_response and st.button("🛠️ Generate Solutions"):
            with st.spinner("Generating solutions to mitigate legal risks..."):
                context = st.session_state.analysis_response.get("context", "")
                solutions = st.session_state.legal_ai.generate_solutions(context)
                st.subheader("✅ Risk Mitigation Suggestions")
                st.markdown(solutions, unsafe_allow_html=True)

    else:
        st.warning("Please load documents first (sidebar → 'Load Documents').")

#Calling main loop
if __name__ == "__main__":
    main()
