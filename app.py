import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import csv

from pipeline import run_pipeline

st.set_page_config(
    page_title="Machine Learning Classification Dashboard",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Machine Learning Classification Dashboard")

st.markdown("""
This dashboard automatically analyzes the **Bank Marketing Dataset**.

You can also upload your own dataset below.
""")

st.divider()

# =====================================================
# DEFAULT BANK DATASET
# =====================================================

BANK_DATASET = "bank+marketing/bank/bank-full.csv"
bank_target = "y"

bank_df = pd.read_csv(
    BANK_DATASET,
    sep=";"
)

st.success("✅ Default Dataset: Bank Marketing")

st.subheader("📂 Dataset Preview")

st.dataframe(
    bank_df.head(),
    use_container_width=True
)

c1, c2, c3 = st.columns(3)

c1.metric("Rows", bank_df.shape[0])
c2.metric("Columns", bank_df.shape[1])
c3.metric(
    "Missing Values",
    int(bank_df.isnull().sum().sum())
)

with st.spinner("Training models..."):

    (
        models,
        results_df,
        reports,
        X_test,
        y_test,
        preprocessor,
        target_encoder
    ) = run_pipeline(
        bank_df,
        bank_target
    )

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

best_model = results_df.iloc[0]

m1, m2, m3 = st.columns(3)

m1.metric(
    "🏆 Best Model",
    best_model["Model"]
)

m2.metric(
    "Accuracy",
    f"{best_model['Accuracy']:.2%}"
)

m3.metric(
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

            im = ax.imshow(
                cm,
                cmap="Blues"
            )

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

    csv_data = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Results CSV",
        data=csv_data,
        file_name="bank_model_results.csv",
        mime="text/csv",
        use_container_width=True
    )

st.divider()

st.header("📤 Analyze Your Own Dataset (Optional)")

st.info(
    "The Bank Marketing dataset has already been analyzed above.\n\n"
    "Upload your own CSV to analyze another dataset."
)

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

# =====================================================
# CUSTOM DATASET
# =====================================================

if uploaded_file is not None:

    uploaded_file.seek(0)
    sample = uploaded_file.read(2048).decode("utf-8")
    uploaded_file.seek(0)

    try:
        delimiter = csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        delimiter = ","

    custom_df = pd.read_csv(
        uploaded_file,
        sep=delimiter
    )

    st.success("✅ Custom Dataset Loaded")

    st.subheader("📂 Uploaded Dataset Preview")

    st.dataframe(
        custom_df.head(),
        use_container_width=True
    )

    u1, u2, u3 = st.columns(3)

    u1.metric("Rows", custom_df.shape[0])
    u2.metric("Columns", custom_df.shape[1])
    u3.metric(
        "Missing Values",
        int(custom_df.isnull().sum().sum())
    )

    custom_target = st.selectbox(
        "🎯 Select Target Column",
        custom_df.columns
    )

    train_uploaded = st.button(
        "🚀 Train Models",
        use_container_width=True
    )

    if train_uploaded:

        with st.spinner("Training models on uploaded dataset..."):

            (
                models,
                results_df,
                reports,
                X_test,
                y_test,
                preprocessor,
                target_encoder
            ) = run_pipeline(
                custom_df,
                custom_target
            )

        results_df = results_df.sort_values(
            by="Accuracy",
            ascending=False
        )

        best_model = results_df.iloc[0]

        a, b, c = st.columns(3)

        a.metric("🏆 Best Model", best_model["Model"])
        b.metric("Accuracy", f"{best_model['Accuracy']:.2%}")
        c.metric("Models Trained", len(results_df))

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

                    im = ax.imshow(
                        cm,
                        cmap="Blues"
                    )

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

            csv_data = results_df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_data,
                file_name="uploaded_model_results.csv",
                mime="text/csv",
                use_container_width=True
            )

st.divider()

st.caption(
    "Developed by Bhavya Sri | Machine Learning Classification Dashboard"
)