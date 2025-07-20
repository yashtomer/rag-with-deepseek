import streamlit as st

from rag_pipeline import answer_query, retrieve_docs, llm


upload = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

user_query = st.text_area("Enter your prompt", height=150, placeholder="What do you want to know?")

ask_question = st.button("Ask AI Lawyer")

if ask_question:
    
    if upload:
        
        st.chat_message("user").write(user_query)

        #RAG Pipeline
        retrieved_docs = retrieve_docs(user_query)
        response = answer_query(retrieved_docs, model=llm, query=user_query)
        st.chat_message("AI Lawyer").write(response)

        # Placeholder response
        #fixed_response = "This is a placeholder response. The AI Lawyer will answer your question based on the uploaded PDF."
        #st.chat_message("AI Lawyer").write(fixed_response)

    else:
        st.error("Please upload a PDF document to proceed.")
else:
    st.info("Please upload a PDF document and enter your question to get started.")
#st.markdown("### Instructions")
#st.markdown("1. Upload a PDF document containing legal information.")
#st.markdown("2. Enter your question in the text area.")
#st.markdown("3. Click 'Ask AI Lawyer' to get a response based on the uploaded document.")
#st.markdown("4. If you have any issues, please ensure the PDF is properly formatted and contains text that can be processed.")
#st.markdown("5. For best results, ensure the PDF is not encrypted and contains clear, readable text.")
#st.markdown("6. If the AI Lawyer does not respond as expected, try re-uploading the PDF or rephrasing your question.")
#st.markdown("7. For any further assistance, please contact support.")
#st.markdown("### Note")
#st.markdown("This is a prototype AI Lawyer application. The responses are generated based on the  uploaded PDF and may not always be accurate. Always consult a qualified legal professional for serious legal matters.")
#st.markdown("### Disclaimer")
#st.markdown("This application is for informational purposes only and does not constitute legal advice. Use  at your own risk. The developers are not responsible for any legal consequences arising from the use of this application.")
#st.markdown("### Contact")
#st.markdown("For any questions or support, please contact us at support@ailawyer.com")

