import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Judul halaman
st.title("Segmentasi Provinsi Berdasarkan Kelayakan Pendidikan")

# Upload file CSV
uploaded_file = st.file_uploader("Upload file CSV dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Hapus kolom kosong
    if 'Unnamed: 14' in df.columns:
        df = df.drop(columns=["Unnamed: 14"])

    st.subheader("Data Awal")
    st.dataframe(df)

    provinsi = df["Provinsi"]
    data_numerik = df.drop(columns=["Provinsi"])

    # Normalisasi
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numerik)

    # Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    # Tambah ke dataframe
    df["Cluster"] = labels

    # Tampilkan hasil cluster
    st.subheader("Hasil Clustering")
    st.dataframe(df[["Provinsi", "Cluster"]])

    # Visualisasi: scatter plot
    st.subheader("Visualisasi Clustering ")
    x_axis = st.selectbox("Pilih fitur X", options=data_numerik.columns)
    y_axis = st.selectbox("Pilih fitur Y", options=data_numerik.columns, index=1)

    fig, ax = plt.subplots()
    scatter = ax.scatter(data_numerik[x_axis], data_numerik[y_axis], c=labels, cmap='viridis')
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_title("Clustering Provinsi")
    st.pyplot(fig)
else:
    st.info("Silakan upload file CSV terlebih dahulu.")
