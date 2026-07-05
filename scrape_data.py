"Script Buat scriping data teori konspirasi dari wikipedia."
"hasilnya disimpan dengan format txt di folder data"

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
    "Denver International Airport conspiracy theories",
    "Roswell UFO incident",
    "Big Pharma conspiracy theories",
    "Fluoridation conspiracy theories",
    "Vaccine Hesitancy",
    "Deep State (conspiracy theory)",
    "Paul Is Dead conspiracy theory",
    "Georgia Guidestones",
    "Simulation hypothesis",

]

def scrape_topic(topic):
    try:
        # Ambil halaman Wikipedia untuk topik tertentu
        page = wikipedia.page(topic, auto_suggest=False)
        content = page.content
        title = page.title
        url = page.url
        return title, url, content
    except wikipedia.exceptions.DisambiguationError as e:
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return page.title, page.url, page.content
        except Exception:
            print(f"[SKIP] Gagal Disambiguasi untuk topik: {topic}")
            return none
    except Exception as e:
        print(f"[ERROR] Gagal mengambil data untuk topik: {topic}. Error: {e}")
        return None
    

def clean_filename(name):
    # Bersihkan nama file dari karakter yang tidak valid
    return "".join(c if c.isalnum() or c in (' ', '.', '_') else '_' for c in name).rstrip()

def main():
    print(f"Mulai Scraping {len(topics)} topik teori konspirasi dari Wikipedia...")

    for i, topic in enumerate(topics,1):
        print(f"[{i}/{len(topics)}] Scraping topik: {topic}")
        result = scrape_topic(topic)

        if result:
            title, url, content = result
            filename = clean_filename(title) + ".txt"
            filepath = os.path.join(DATA_DIR, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"Title: {title}\n")
                f.write(f"Sumber: {url}\n\n")
                f.write("=" * 50 + "\n\n")
                f.write(content)

            print(f"Sukse Tersimpan : {filepath} ({len(content)} karakter)")
        
        time.sleep(1)  # Delay untuk menghindari rate limit

    print("Selesai scraping semua topik.")

if __name__ == "__main__":
    main()
    

       