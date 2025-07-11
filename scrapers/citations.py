import mysql.connector
import requests
import time
import csv
import os
import pandas as pd
from datetime import datetime

API_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}
CSV_FILE = "citations.csv"
PROGRESS_FILE = "processed_papers.txt"

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="@26March2002",
        database="scholars_data"
    )

def get_all_papers():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT PID, PTitle FROM PAPERS")
    papers = cursor.fetchall()
    cursor.close()
    conn.close()
    return papers

def get_paper_id_by_title(title):
    try:
        query = requests.utils.quote(title)
        url = f"{API_BASE}/paper/search?query={query}&limit=1&fields=paperId"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        if data.get("data"):
            return data["data"][0].get("paperId")
    except Exception as e:
        print(f"❌ Error fetching paper ID for title '{title}': {e}")
    return None

def get_citations(paper_id, max_citations=100):
    citations = []
    offset = 0
    limit = 100
    while len(citations) < max_citations:
        url = f"{API_BASE}/paper/{paper_id}/citations?limit={limit}&offset={offset}&fields=title,authors,year,venue,paperId"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            batch = data.get("data", [])
            if not batch:
                break
            for item in batch:
                cited = item.get("citingPaper", {})
                citations.append({
                    "CID": cited.get("paperId", ""),
                    "CTitle": cited.get("title", ""),
                    "CAuthors": ", ".join([a.get("name", "") for a in cited.get("authors", [])]),
                    "CYear": cited.get("year"),
                    "CPublisher": cited.get("venue", "")
                })
            offset += limit
            if len(batch) < limit:
                break
            time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error fetching citations for paper ID {paper_id}: {e}")
            break
    return citations

def citation_exists(cursor, pid, cid):
    query = "SELECT 1 FROM CITATIONS WHERE PID = %s AND CID = %s LIMIT 1"
    cursor.execute(query, (pid, cid))
    return cursor.fetchone() is not None

def insert_citation(cursor, pid, citation):
    if not citation["CID"]:
        return False
    if citation_exists(cursor, pid, citation["CID"]):
        return False
    query = """
        INSERT INTO CITATIONS (PID, CID, CTitle, CAuthors, CYear, CPublisher)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
    
        citation["CID"],
        pid,
        citation["CTitle"][:500],
        citation["CAuthors"][:500],
        citation["CYear"],
        citation["CPublisher"][:255],
    ))
    return True

def export_citation_to_csv_single(writer, citation, file_handle):
    writer.writerow(citation)
    file_handle.flush()  # flush immediately after writing

def load_processed_papers():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f.readlines())

def save_processed_paper(pid):
    with open(PROGRESS_FILE, "a", encoding="utf-8") as f:
        f.write(f"{pid}\n")

def clean_citations_csv():
    print("🧹 Cleaning citations.csv...")
    try:
        df = pd.read_csv(CSV_FILE, dtype=str)
        original_len = len(df)
        df.dropna(subset=[ "CID","PID","CTitle", "CAuthors", "CYear"], inplace=True)
        df = df[df["PID"].str.strip() != ""]
        df = df[df["CID"].str.strip() != ""]
        df = df[df["CTitle"].str.strip() != ""]
        cleaned_len = len(df)
        df.to_csv(CSV_FILE, index=False, encoding="utf-8")
        print(f"✅ Cleaned {original_len - cleaned_len} rows with missing values.")
    except Exception as e:
        print(f"❌ Error cleaning CSV: {e}")

def main():
    clean_citations_csv()

    papers = get_all_papers()
    print(f"📄 Found {len(papers)} papers.\n")

    processed_papers = load_processed_papers()
    print(f"🗂️ Already processed {len(processed_papers)} papers.\n")

    conn = connect_db()
    cursor = conn.cursor()

    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["PID", "CID", "CTitle", "CAuthors", "CYear", "CPublisher"])
        if not file_exists:
            writer.writeheader()

        try:
            for paper in papers:
                pid = str(paper["PID"])
                title = paper["PTitle"]

                if pid in processed_papers:
                    print(f"➡️ Skipping already processed paper PID={pid}")
                    continue

                print(f"➡️ Processing PID={pid}: '{title[:60]}...'")

                ss_paper_id = get_paper_id_by_title(title)
                if not ss_paper_id:
                    print(f"❌ No Semantic Scholar ID found for: {title}")
                    save_processed_paper(pid)
                    continue

                citations = get_citations(ss_paper_id)
                print(f"🔸 Found {len(citations)} citations externally.")

                inserted = 0
                for c in citations:
                    try:
                        if insert_citation(cursor, pid, c):
                            c["PID"] = pid
                            export_citation_to_csv_single(writer, c, f)
                            inserted += 1
                    except Exception as e:
                        print(f"⚠️ Failed to insert citation: {e}")

                conn.commit()
                save_processed_paper(pid)
                print(f"✅ Inserted {inserted} new citations for PID={pid}\n")

        except Exception as main_e:
            print(f"❗ Unexpected error: {main_e}")
        finally:
            cursor.close()
            conn.close()

if __name__ == "__main__":
    main()
