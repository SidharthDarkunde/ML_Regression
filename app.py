# import streamlit as st
# import joblib
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import time

# # Page config
# st.set_page_config(
#     page_title="CGPA to Package Predictor",
#     page_icon="🎓",
#     layout="centered"
# )

# # Load model
# model = joblib.load("regression_model.joblib")

# # Title
# st.markdown("<h1 style='text-align:center;'>🎓 CGPA → Package Predictor</h1>", unsafe_allow_html=True)
# st.markdown("<p style='text-align:center;'>Visual prediction of job package using ML</p>", unsafe_allow_html=True)
# st.divider()

# # Input
# st.subheader("📌 Enter Your CGPA")
# cgpa = st.slider("Select CGPA", 0.0, 10.0, 7.0, 0.1)

# # Predict
# if st.button("🚀 Predict Package"):
#     with st.spinner("Running ML model..."):
#         time.sleep(1)
#         X = np.array([[cgpa]])
#         prediction = model.predict(X)
#         pred_value = float(prediction[0])

#     st.success("Prediction Completed 🎉")

#     # ---- METRIC ----
#     st.metric("💰 Expected Package (LPA)", f"{pred_value:.2f}")

#     st.divider()

#     # ---- CHART 1: CGPA vs Package LINE ----
#     st.subheader("📈 CGPA vs Predicted Package")

#     cgpa_range = np.linspace(0, 10, 50).reshape(-1, 1)
#     package_range = model.predict(cgpa_range)

#     fig1, ax1 = plt.subplots()
#     ax1.plot(cgpa_range, package_range)
#     ax1.scatter(cgpa, pred_value)
#     ax1.set_xlabel("CGPA")
#     ax1.set_ylabel("Package (LPA)")
#     ax1.set_title("Regression Line")

#     st.pyplot(fig1)

#     # ---- CHART 2: BAR CHART ----
#     st.subheader("📊 Your Package Overview")

#     df = pd.DataFrame({
#         "Category": ["Predicted Package"],
#         "LPA": [pred_value]
#     })

#     fig2, ax2 = plt.subplots()
#     ax2.bar(df["Category"], df["LPA"])
#     ax2.set_ylabel("LPA")
#     ax2.set_title("Package Prediction")

#     st.pyplot(fig2)

#     # ---- FEEDBACK ----
#     if pred_value >= 20:
#         st.info("🔥 Excellent profile! Top-tier opportunities.")
#     elif pred_value >= 10:
#         st.info("👍 Good profile. Keep upgrading skills.")
#     else:
#         st.info("📘 Improve CGPA & projects for better offers.")

# # Footer
# st.divider()
# st.caption("📊 Machine Learning Regression | Streamlit App")



# import streamlit as st
# import joblib
# import numpy as np

# # Load trained model
# model = joblib.load("regression_model.joblib")

# st.title("CGPA → Package Predictor")
# st.write("Enter your CGPA to predict expected package")

# cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, step=0.1)

# if st.button("Predict Package"):
#     cgpa_array = np.array([[cgpa]])
#     prediction = model.predict(cgpa_array)
    
#     pred_value = float(prediction[0])   
#     st.success(f"Predicted Package: {pred_value:.2f} LPA")



import streamlit as st
import joblib
import numpy as np
import time

# Page config
st.set_page_config(
    page_title="CGPA to Package Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load trained model
model = joblib.load("regression_model.joblib")

# Title & description
st.markdown("<h1 style='text-align: center;'>🎓 CGPA → Package Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Predict your expected job package based on CGPA</p>",
    unsafe_allow_html=True
)

st.divider()

# Input section
st.subheader("📌 Enter Academic Details")

cgpa = st.slider(
    "Select your CGPA",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.1
)

# Predict button
if st.button("🚀 Predict Package"):
    
    with st.spinner("Predicting package..."):
        time.sleep(1)
        cgpa_array = np.array([[cgpa]])
        prediction = model.predict(cgpa_array)
        pred_value = float(prediction[0])

    st.success("Prediction Successful! 🎉")

    # Display result
    st.metric(
        label="💰 Expected Package (LPA)",
        value=f"{pred_value:.2f}"
    )

    # Progress bar visualization
    progress = min(pred_value / 50, 1.0)  # scale safely
    st.progress(progress)

    # Friendly message
    if pred_value >= 20:
        st.info("🔥 Excellent! Top-tier package potential.")
    elif pred_value >= 10:
        st.info("👍 Good! Strong placement chances.")
    else:
        st.info("📘 Keep improving skills and CGPA!")

# Footer
st.divider()
st.caption("📊 ML Regression Model | Built with Streamlit")
