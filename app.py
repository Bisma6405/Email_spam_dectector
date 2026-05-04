import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)

# ──────────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("spam_detector.pkl", "rb") as f:
        return pickle.load(f)

bundle       = load_model()
model        = bundle['model']
feature_cols = bundle['feature_cols']
label_map    = bundle['label_map']
# scaler removed — not needed
# ──────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────────
def predict_email(word_count_dict: dict) -> dict:
    vec   = np.array([[word_count_dict.get(w, 0) for w in feature_cols]])
    label = model.predict(vec)[0]
    prob  = model.predict_proba(vec)[0][1]
    return {
        'label': label_map[label],
        'spam_probability': round(float(prob), 4),
        'ham_probability':  round(1 - float(prob), 4),
    }

def text_to_word_counts(text: str) -> dict:
    """Convert raw email text to word-count dict matching training features."""
    import re
    words = re.findall(r'[a-zA-Z]+', text.lower())
    counts = {}
    for w in words:
        if w in feature_cols:
            counts[w] = counts.get(w, 0) + 1
    return counts

# ──────────────────────────────────────────────────────────────
# UI — HEADER
# ──────────────────────────────────────────────────────────────
st.title("📧 Email Spam Detector")
st.markdown("Paste the content of any email below and the model will classify it as **Spam** or **Ham**.")
st.divider()

# ──────────────────────────────────────────────────────────────
# UI — INPUT
# ──────────────────────────────────────────────────────────────
email_input = st.text_area(
    label="✉️ Paste Email Content Here",
    placeholder="e.g. Congratulations! You've won a FREE prize. Click here now...",
    height=200
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔍 Analyze Email", use_container_width=True, type="primary")

# ──────────────────────────────────────────────────────────────
# UI — RESULT
# ──────────────────────────────────────────────────────────────
if predict_btn:
    if not email_input.strip():
        st.warning("⚠️ Please paste some email content first.")
    else:
        word_counts = text_to_word_counts(email_input)
        result      = predict_email(word_counts)

        st.divider()

        is_spam = result['label'] == 'spam'

        # Result banner
        if is_spam:
            st.error("🚨 **SPAM DETECTED**", icon="🚫")
        else:
            st.success("✅ **LEGITIMATE EMAIL (Ham)**", icon="✉️")

        # Probability meters
        st.markdown("### Confidence Scores")
        col_spam, col_ham = st.columns(2)

        with col_spam:
            st.metric(
                label="🔴 Spam Probability",
                value=f"{result['spam_probability']*100:.1f}%"
            )
            st.progress(result['spam_probability'])

        with col_ham:
            st.metric(
                label="✅ Ham Probability",
                value=f"{result['ham_probability']*100:.1f}%"
            )
            st.progress(result['ham_probability'])

        # Word match stats
        matched = {w: c for w, c in word_counts.items() if c > 0}
        st.divider()
        st.markdown(f"### 📊 Analysis Details")

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            st.metric("Words in Email",   len(email_input.split()))
            st.metric("Features Matched", len(matched))
        with detail_col2:
            st.metric("Model Used", bundle.get('model_name', 'Ensemble'))
            st.metric("Vocabulary Size", len(feature_cols))

        # Top matched words
        if matched:
            st.markdown("**Top matched words from vocabulary:**")
            top_words = sorted(matched.items(), key=lambda x: x[1], reverse=True)[:10]
            words_df  = pd.DataFrame(top_words, columns=["Word", "Count"])
            st.dataframe(words_df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────
# UI — SAMPLE EMAILS SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🧪 Try Sample Emails")
    st.markdown("Click a sample to auto-fill the input box.")

    samples = {
        "🔴 Spam Sample 1": (
            "Congratulations! You have been selected to receive a FREE gift card worth $500. "
            "Click here now to claim your prize. Limited time offer! Buy online today. "
            "Visit our website http://free-prizes.com to get your reward. "
            "This is not a spam message. Unsubscribe anytime."
        ),
        "🔴 Spam Sample 2": (
            "URGENT: Your account will be suspended. Verify your information immediately. "
            "Buy cheap medications online. Viagra, Cialis, pills at lowest price. "
            "Click the link below. Free shipping worldwide. Money back guarantee. "
            "Order now and save! Visit http://cheapmeds.net"
        ),
        "✅ Ham Sample 1": (
            "Hi, just wanted to confirm our meeting tomorrow at 2pm. "
            "Please review the attached report before the call. "
            "Let me know if you have any questions. Thanks, John."
        ),
        "✅ Ham Sample 2": (
            "The quarterly sales report has been forwarded to the team. "
            "Please review the enron contract details and gas volume nominations. "
            "Let us know if any changes are needed before the deadline on Friday."
        ),
    }

    for label, text in samples.items():
        if st.button(label, use_container_width=True):
            st.session_state['sample_text'] = text
            st.rerun()

    # Inject sample into text area via session state
    if 'sample_text' in st.session_state:
        email_input = st.session_state['sample_text']
        del st.session_state['sample_text']

    st.divider()
    st.markdown("**Model Info**")
    st.info(
        f"**Name:** {bundle.get('model_name','Ensemble')}\n\n"
        f"**Vocabulary:** {len(feature_cols)} words\n\n"
        f"**Labels:** Ham (0) / Spam (1)"
    )