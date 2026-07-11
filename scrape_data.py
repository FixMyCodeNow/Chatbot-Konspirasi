"Script Buat scriping data teori konspirasi dari wikipedia."
"hasilnya disimpan dengan format txt di folder data"

from matplotlib.pyplot import title
import wikipedia
import os
import time

wikipedia.set_lang("en")  # Set bahasa ke Inggris karena dataset lebih banyak
#folder buat menyimpan hasil scraping
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

#Daftar topik teori konspirasi yang ingin diambil datanya
topics = [
    "Flat Earth conspiracy theory",
    "Moon landing conspiracy theories",
    "Illuminati",
    "New World Order (conspiracy theory)",
    "Chemtrail conspiracy theory",
    "Area 51",
    "JFK assassination conspiracy theories",
    "9/11 conspiracy theories",
    "Reptilian conspiracy theory",
    "QAnon",
    "Bermuda Triangle",
    "Denver International Airport",
    "Roswell UFO incident",
    "Big Pharma conspiracy theories",
    "Water fluoridation controversy",
    "Vaccine hesitancy",
    "Deep state in the United States",
    "Dyatlov Pass incident",
    "Georgia Guidestones",
    "Simulation hypothesis",

    # 10 topik baru yang mau ditambahin:
    "Bilderberg Group",
    "Pizzagate conspiracy theory",
    "Sandy Hook Elementary School shooting conspiracy theories",
    "Project MKUltra",
    "HAARP",
    "Death of Diana, Princess of Wales conspiracy theories",
    "Barack Obama citizenship conspiracy theories",
    "Bohemian Grove",
    "Operation Northwoods",
]

def scrape_topic(topic, max_retries=5):
    """Ambil isi artikel Wikipedia untuk satu topik, dengan retry kalau gagal."""
    for attempt in range(max_retries):
        try:
            page = wikipedia.page(topic, auto_suggest=False)
            return page.title, page.content, page.url
        except wikipedia.exceptions.DisambiguationError as e:
            try:
                page = wikipedia.page(e.options[0], auto_suggest=False)
                return page.title, page.content, page.url
            except Exception:
                print(f'[SKIP] Gagal Disambiguasi: {topic}')
                return None
        except wikipedia.exceptions.PageError:
            print(f'[SKIP] Halaman tidak ditemukan: {topic}')
            return None
        except Exception as e:
            print(f'[RETRY {attempt + 1} / {max_retries}] Gagal ambil topik: {topic}. {e}')
            time.sleep(5)  # Tunggu sebelum retry

    print(f"[SKIP] Gagal total setelah {max_retries} percobaan: {topic}")
    return None

def clean_filename(name):
    """Bersihkan judul artikel untuk dijadikan nama file yang valid."""
    return "".join(c if c.isalnum() or c in (" ", "_") else "_" for c in name).strip()

def main():
    print(f"Mulai Scraping {len(topics)} topik ... \n")

    for i, topic in enumerate(topics, 1):
        # Cek apakah topik ini kemungkinan sudah pernah di-scrape sebelumnya
        existing_files = os.listdir(DATA_DIR)
        keyword = topic.split(" (")[0].split(",")[0].lower()
        already_scraped = any(keyword in f.lower() for f in existing_files)

        if already_scraped:
            print(f"{i}/{len(topics)} [SKIP] Topik '{topic}' kemungkinan sudah di-scrape sebelumnya")
            continue
        print(f"{i}/{len(topics)} Scraping topik: '{topic}' ...")
        result = scrape_topic(topic)

        if result:
            title, content, url = result
            filename = clean_filename(title) + ".txt"
            filepath = os.path.join(DATA_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Title: {title}\n")
                f.write(f"Sumber: {url}\n\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)

            print(f" [SUKSES] Disimpan ke: {filepath} ({len(content)} karakter)")

        time.sleep(5)  # Delay biar gak terlalu cepat

    print("Selesai Semua data ada di folder 'data'.")


if __name__ == "__main__":
    main()
    

       