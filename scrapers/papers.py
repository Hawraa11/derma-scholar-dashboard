import mysql.connector
import requests
import csv
import time
import re
import os

API_BASE = "https://api.semanticscholar.org/graph/v1"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MAX_PAPERS = 1000
CSV_FILE = "papers.csv"

# Connect to MySQL
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="@26March2002",
        database="scholars_data"
    )

# Get researchers
def get_researchers():
    conn = connect_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT RID, RName FROM RESEARCHERS")
    researchers = cursor.fetchall()
    cursor.close()
    conn.close()
    return researchers

# Semantic Scholar author ID
def get_author_id(name):
    try:
        queries = [name, name.split()[-1], re.sub(r"\b\w\b\.?", "", name).strip()]
        for query in queries:
            url = f"{API_BASE}/author/search?query={requests.utils.quote(query)}&limit=3"
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            if data.get("data"):
                return data["data"][0].get("authorId")
    except Exception as e:
        print(f"Error fetching author ID for '{name}': {e}")
    return None

# Fetch papers by author ID
def fetch_papers_for_author(author_id, max_papers=100):
    papers = []
    offset = 0
    limit = 100
    while len(papers) < max_papers:
        to_fetch = min(limit, max_papers - len(papers))
        url = f"{API_BASE}/author/{author_id}/papers?limit={to_fetch}&offset={offset}&fields=paperId,title,authors,year,url,publicationTypes,venue,citationCount"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            batch = data.get("data", [])
            if not batch:
                break
            papers.extend(batch)
            offset += to_fetch
            if len(batch) < to_fetch:
                break
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching papers for author {author_id}: {e}")
            break
    return papers[:max_papers]

# Fallback: search by author name
def fetch_papers_by_author_name(name, max_papers=100):
    papers = []
    offset = 0
    limit = 100
    while len(papers) < max_papers:
        to_fetch = min(limit, max_papers - len(papers))
        query = requests.utils.quote(f'author:"{name}"')
        url = f"{API_BASE}/paper/search?query={query}&limit={to_fetch}&offset={offset}&fields=paperId,title,authors,year,url,publicationTypes,venue,citationCount"
        try:
            response = requests.get(url, headers=HEADERS)
            response.raise_for_status()
            data = response.json()
            batch = data.get("data", [])
            if not batch:
                break
            papers.extend(batch)
            offset += to_fetch
            if len(batch) < to_fetch:
                break
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching papers by author name '{name}': {e}")
            break
    return papers[:max_papers]

# Check if paper already exists
def get_existing_paper_pid(cursor, rid, title, link):
    query = "SELECT PID FROM PAPERS WHERE RID = %s AND (PTitle = %s OR PLink = %s) LIMIT 1"
    cursor.execute(query, (rid, title, link))
    result = cursor.fetchone()
    return result[0] if result else None

# Insert paper into DB
def insert_paper(cursor, paper, rid):
    title = paper.get("title", "")[:255]
    authors = ", ".join([a.get("name", "") for a in paper.get("authors", [])])
    year = paper.get("year")
    year_val = year if year and year > 0 else None
    link = paper.get("url")
    ptype_raw = paper.get("publicationTypes")
    ptype = ", ".join(ptype_raw)[:50] if isinstance(ptype_raw, list) else (ptype_raw or "Unknown")[:50]
    source = paper.get("venue", "")[:150]
    citations = paper.get("citationCount", 0)

    existing_pid = get_existing_paper_pid(cursor, rid, title, link)
    if existing_pid:
        return existing_pid

    insert_query = """
        INSERT INTO PAPERS (RID, PTitle, PAuthors, PYear, PLink, PType, PSource, PCitations)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(insert_query, (rid, title, authors, year_val, link, ptype, source, citations))
    return cursor.lastrowid

# Export to CSV
def export_to_csv(papers_data):
    keys = ["PID", "RID", "PTitle", "PAuthors", "PYear", "PLink", "PType", "PSource", "PCitations"]
    existing_entries = set()
    temp_rows = []

    if os.path.exists(CSV_FILE):
        with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            has_pid = "PID" in reader.fieldnames
            for row in reader:
                pid_val = row.get("PID", "") if has_pid else ""
                temp_rows.append({
                    "PID": pid_val,
                    "RID": row.get("RID", ""),
                    "PTitle": row.get("PTitle", ""),
                    "PAuthors": row.get("PAuthors", ""),
                    "PYear": row.get("PYear", ""),
                    "PLink": row.get("PLink", ""),
                    "PType": row.get("PType", ""),
                    "PSource": row.get("PSource", ""),
                    "PCitations": row.get("PCitations", "")
                })
                key = (pid_val, row.get("PTitle", ""), row.get("PLink", ""))
                existing_entries.add(key)

        if not has_pid:
            with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                for row in temp_rows:
                    writer.writerow(row)

    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if os.path.getsize(CSV_FILE) == 0:
            writer.writeheader()
        for paper in papers_data:
            key = (str(paper.get("PID", "")), paper["PTitle"], paper["PLink"])
            if key not in existing_entries:
                writer.writerow(paper)

# Cleanup and fix missing PIDs
def fix_and_clean_csv():
    if not os.path.exists(CSV_FILE):
        return

    conn = connect_db()
    cursor = conn.cursor(dictionary=True)

    cleaned_rows = []
    fixed_count = 0
    deleted_count = 0

    with open(CSV_FILE, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        keys = reader.fieldnames

    def normalize_title(title):
        return re.sub(r'\s+', ' ', title).strip().lower() if title else ""

    for row in rows:
        pid = row.get("PID", "").strip()
        title = row.get("PTitle", "").strip()
        link = row.get("PLink", "").strip()
        rid = row.get("RID")

        if pid:
            cleaned_rows.append(row)
            continue

        if not title and not link:
            deleted_count += 1
            continue

        found_pid = None

        if link:
            cursor.execute("SELECT PID FROM PAPERS WHERE RID = %s AND PLink = %s LIMIT 1", (rid, link))
            result = cursor.fetchone()
            if result:
                found_pid = result["PID"]

        if not found_pid and title:
            norm_title = normalize_title(title)
            cursor.execute("SELECT PID, PTitle FROM PAPERS WHERE RID = %s", (rid,))
            for paper in cursor.fetchall():
                if normalize_title(paper["PTitle"]) == norm_title:
                    found_pid = paper["PID"]
                    break

        if found_pid:
            row["PID"] = str(found_pid)
            fixed_count += 1
            cleaned_rows.append(row)
        else:
            deleted_count += 1

    cursor.close()
    conn.close()

    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"🛠️ Fixed PIDs for {fixed_count} rows.")
    print(f"🗑️ Deleted {deleted_count} incomplete/unmatched rows.")

# Main
def main():
    researchers = get_researchers()
    print(f"🔍 Found {len(researchers)} researchers.\n")

    conn = connect_db()
    cursor = conn.cursor()
    all_papers = []

    for researcher in researchers:
        rid = researcher["RID"]
        name = researcher["RName"]
        print(f"➡️ Processing: {name} (RID={rid})")

        author_id = get_author_id(name)
        papers = fetch_papers_for_author(author_id, MAX_PAPERS) if author_id else fetch_papers_by_author_name(name, MAX_PAPERS)

        print(f"🔸 Found {len(papers)} papers for {name}")

        inserted_count = 0
        for paper in papers:
            try:
                pid = insert_paper(cursor, paper, rid)
                if pid:
                    inserted_count += 1
                    print(f"✅ Inserted/Found PID {pid}: {paper.get('title','')[:60]}...")
                all_papers.append({
                    "PID": pid,
                    "RID": rid,
                    "PTitle": paper.get("title", ""),
                    "PAuthors": ", ".join([a.get("name", "") for a in paper.get("authors", [])]),
                    "PYear": paper.get("year"),
                    "PLink": paper.get("url"),
                    "PType": ", ".join(paper.get("publicationTypes", [])) if isinstance(paper.get("publicationTypes"), list) else (paper.get("publicationTypes") or "Unknown"),
                    "PSource": paper.get("venue", ""),
                    "PCitations": paper.get("citationCount", 0)
                })
            except Exception as e:
                print(f"⚠️ Failed to insert paper: {e}")

        conn.commit()
        print(f"ℹ️ Inserted/Found {inserted_count} papers for {name}\n")

    cursor.close()
    conn.close()

    print(f"📁 Exporting all papers to {CSV_FILE} ...")
    export_to_csv(all_papers)
    fix_and_clean_csv()
    print("✅ Export and cleanup completed.")

if __name__ == "__main__":
    main()
