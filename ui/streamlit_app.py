import requests, streamlit as st

st.title("Financial News Chat (Local JSON)")
q = st.text_input("Ask about a ticker or topic (e.g., 'What’s up with AAPL earnings?')")
use_llm = st.checkbox("Polish with Together AI", value=False)
k = st.slider("Results", 1, 5, 3)

if st.button("Ask") and q:
    r = requests.post("http://127.0.0.1:8000/chat", json={"message": q, "k": k, "use_llm": use_llm})
    data = r.json()["answers"]
    for a in data:
        st.markdown(f"**{a['ticker']} — {a['title']}**")
        st.write(a["summary"])
        if a["link"]:
            st.markdown(f"[Open source]({a['link']})")
        st.caption(f"score={a['score']:.2f}")