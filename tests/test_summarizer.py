from app.summarizer import summarize

def test_summary_short():
    s = summarize(["Apple reported earnings.", "Revenue grew.", "Guidance was raised."])
    assert 0 < len(s) <= 500