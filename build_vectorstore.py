import os
import time
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def build_vectorstore(google_api_key, data_dir="data", index_dir="faiss_index"):
    os.environ["GOOGLE_API_KEY"] = google_api_key

    print("Loading documents from directory...")
    loader = DirectoryLoader(
        data_dir,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    documents = loader.load()
    print(f"Berhasil load {len(documents)} dokumen.")

    print("Memecah dokumen menjadi chunk...")
    # chunk_size diperbesar biar jumlah chunk berkurang (hemat kuota harian)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunk yang dihasilkan: {len(chunks)}")

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    batch_size = 20
    vectorstore = None
    start_batch = 0

    # Kalau index sudah ada sebagian (dari proses sebelumnya yang kepotong), lanjutkan
    progress_file = os.path.join(index_dir, "progress.txt")
    if os.path.exists(index_dir) and os.path.exists(progress_file):
        print("Ditemukan progress sebelumnya, melanjutkan...")
        vectorstore = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        with open(progress_file, "r") as f:
            start_batch = int(f.read().strip())
        print(f"Melanjutkan dari batch {start_batch + 1}")

    total_batches = (len(chunks) - 1) // batch_size + 1

    for i in range(start_batch * batch_size, len(chunks), batch_size):
        batch_num = i // batch_size
        batch = chunks[i:i + batch_size]
        print(f"Processing batch {batch_num + 1} / {total_batches}...")

        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)

            # Simpan progress setiap habis 1 batch, biar kalau kepotong gak hilang
            os.makedirs(index_dir, exist_ok=True)
            vectorstore.save_local(index_dir)
            with open(progress_file, "w") as f:
                f.write(str(batch_num + 1))

            time.sleep(15)

        except Exception as e:
            print(f"\nBerhenti di batch {batch_num + 1} karena error: {e}")
            print("Progress udah kesimpen. Jalanin ulang script ini nanti buat lanjut dari sini.")
            return

    # Kalau semua selesai, hapus file progress
    if os.path.exists(progress_file):
        os.remove(progress_file)

    print(f"\nSelesai! Vector database lengkap tersimpan di folder '{index_dir}/'")

if __name__ == "__main__":
    api_key = input("Masukkan Google API Key: ").strip()
    build_vectorstore(api_key)