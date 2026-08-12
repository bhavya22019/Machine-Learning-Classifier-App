import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pipeline import run_pipeline

st.set_page_config(
    page_title="Machine Learning Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Classification Dashboard")

st.markdown("""
Upload a dataset, train multiple machine learning models,
compare their performance, and identify the best model.
""")

st.divider()

with st.sidebar:

    st.header("⚙️ Configuration")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file, sep=";")

    with st.sidebar:

        target_column = st.selectbox(
            "🎯 Target Column",
            df.columns
        )

        train = st.button(
            "🚀 Train Models",
            use_container_width=True
        )

    st.subheader("📂 Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    if train:

        with st.spinner("Training models..."):

            (
                models,
                results_df,
                reports,
                X_test,
                y_test,
                preprocessor,
                target_encoder
            ) = run_pipeline(df, target_column)

            results_df = results_df.sort_values(
                by="Accuracy",
                ascending=False
            )

            best_model = results_df.iloc[0]

            a, b, c = st.columns(3)

            a.metric(
                "🏆 Best Model",
                best_model["Model"]
            )

            b.metric(
                "Accuracy",
                f"{best_model['Accuracy']:.2%}"
            )

            c.metric(
                "Models Trained",
                len(results_df)
            )

            st.divider()

            st.subheader("📊 Model Comparison")

            st.dataframe(
                results_df,
                use_container_width=True
            )

            st.subheader("📈 Accuracy Comparison")

            st.bar_chart(
                results_df.set_index("Model")["Accuracy"]
            )

            tab1, tab2, tab3 = st.tabs(
                [
                    "📊 Reports",
                    "📉 Confusion Matrix",
                    "💾 Downloads"
                ]
            )


            with tab1:

                st.subheader("📊 Classification Reports")

                for model_name, report in reports.items():

                    with st.expander(f"📄 {model_name}"):

                        report_df = pd.DataFrame(
                            report["classification_report"]
                        ).transpose()

                        st.dataframe(
                            report_df,
                            use_container_width=True
                        )

            with tab2:

                st.subheader("📉 Confusion Matrices")

                for model_name, report in reports.items():

                    with st.expander(f"📊 {model_name}"):

                        cm = report["confusion_matrix"]

                        cm_percent = (
                            cm.astype(float)
                            / cm.sum(axis=1)[:, None]
                            * 100
                        )

                        fig, ax = plt.subplots(figsize=(7, 5))

                        im = ax.imshow(cm, cmap="Blues")

                        ax.set_xticks([0, 1])
                        ax.set_yticks([0, 1])

                        ax.set_xticklabels(
                            ["Predicted No", "Predicted Yes"]
                        )

                        ax.set_yticklabels(
                            ["Actual No", "Actual Yes"]
                        )

                        ax.set_xlabel("Predicted Label")
                        ax.set_ylabel("Actual Label")

                        ax.set_title(
                            f"{model_name} Confusion Matrix",
                            fontsize=14,
                            fontweight="bold"
                        )

                        for i in range(2):
                            for j in range(2):

                                ax.text(
                                    j,
                                    i,
                                    f"{cm[i,j]}\n({cm_percent[i,j]:.1f}%)",
                                    ha="center",
                                    va="center",
                                    fontsize=12,
                                    fontweight="bold"
                                )

                        fig.colorbar(im)

                        st.pyplot(fig)

                        plt.close(fig)

            with tab3:

                st.subheader("💾 Download Results")

                csv = results_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name="model_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    st.divider()

    st.caption(
        "Developed by Bhavya Sri | Machine Learning Classification Dashboard"
    )


