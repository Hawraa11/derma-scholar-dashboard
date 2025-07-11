import mysql.connector
from scholarly import scholarly
from urllib.parse import urlparse, parse_qs
import time
import csv

# List of Google Scholar profile URLs
urls = [
        
"https://scholar.google.com/citations?user=tnL6fqQAAAAJ&hl=en&oi=ao" ,
"https://scholar.google.com/citations?user=smHyoHgAAAAJ&hl=en&oi=ao" ,
"https://scholar.google.com/citations?hl=en&user=w2RAT7MAAAAJ",
"https://scholar.google.com/citations?hl=en&user=jxXyDOEAAAAJ",
"https://scholar.google.com/citations?hl=en&user=77GMUVMAAAAJ",
"https://scholar.google.com/citations?hl=en&user=Z5k8rhkAAAAJ",
"https://scholar.google.com/citations?hl=en&user=GCOhGBYAAAAJ",
"https://scholar.google.com/citations?hl=en&user=m8dT-9oAAAAJ",
"https://scholar.google.com/citations?hl=en&user=miMA_4kAAAAJ",
"https://scholar.google.com/citations?hl=en&user=Bm_92fEAAAAJ",
"https://scholar.google.com/citations?hl=en&user=9NYyyxcAAAAJ",
"https://scholar.google.com/citations?hl=en&user=G_jds3IAAAAJ",
"https://scholar.google.com/citations?hl=en&user=n7l2z0AAAAAJ",
"https://scholar.google.com/citations?hl=en&user=EHnRbGoAAAAJ",
"https://scholar.google.com/citations?hl=en&user=j9fwqP0AAAAJ",
"https://scholar.google.com/citations?hl=en&user=K0DxNSkAAAAJ",
"https://scholar.google.com/citations?hl=en&user=prYTAwwAAAAJ",
"https://scholar.google.com/citations?hl=en&user=C1Qy49QAAAAJ",
"https://scholar.google.com/citations?hl=en&user=expI1B4AAAAJ",
"https://scholar.google.com/citations?hl=en&user=8QOziroAAAAJ",
"https://scholar.google.com/citations?hl=en&user=ZMvMov0AAAAJ",
"https://scholar.google.com/citations?hl=en&user=dEUOk14AAAAJ",
"https://scholar.google.com/citations?hl=en&user=S71pCjwAAAAJ",
"https://scholar.google.com/citations?hl=en&user=IHQiEqMAAAAJ",
"https://scholar.google.com/citations?hl=en&user=kFM6dAcAAAAJ",
"https://scholar.google.com/citations?hl=en&user=-Aa3lugAAAAJ",
"https://scholar.google.com/citations?hl=en&user=tTE2J0wAAAAJ",
"https://scholar.google.com/citations?hl=en&user=bNLkELsAAAAJ",
"https://scholar.google.com/citations?hl=en&user=T5dKXjsAAAAJ",
"https://scholar.google.com/citations?hl=en&user=q4TUvuMAAAAJ",
"https://scholar.google.com/citations?hl=en&user=LATQiDcAAAAJ",
"https://scholar.google.com/citations?hl=en&user=gB-Rx5MAAAAJ",
"https://scholar.google.com/citations?hl=en&user=FCbGtqAAAAAJ",
"https://scholar.google.com/citations?hl=en&user=smHyoHgAAAAJ",
"https://scholar.google.com/citations?hl=en&user=zVjBV5AAAAAJ",
"https://scholar.google.com/citations?hl=en&user=orb8dJsAAAAJ",
"https://scholar.google.com/citations?hl=en&user=nQyzVlqOVH8C",
"https://scholar.google.com/citations?hl=en&user=CJOiQpUAAAAJ",
"https://scholar.google.com/citations?hl=en&user=Qd0wN6sAAAAJ",
"https://scholar.google.com/citations?hl=en&user=dqBJgjYAAAAJ",
"https://scholar.google.com/citations?hl=en&user=pFo4Q54AAAAJ",
"https://scholar.google.com/citations?hl=en&user=gTRJQpUAAAAJ",
"https://scholar.google.com/citations?hl=en&user=Sf9qO1QAAAAJ",
"https://scholar.google.com/citations?hl=en&user=K1S0DOAAAAAJ",
"https://scholar.google.com/citations?hl=en&user=UPq0A6YAAAAJ",
"https://scholar.google.com/citations?hl=en&user=8G84aogAAAAJ",
"https://scholar.google.com/citations?hl=en&user=0rWmWAYAAAAJ",
"https://scholar.google.com/citations?hl=en&user=MvVlqyAAAAAJ",
"https://scholar.google.com/citations?hl=en&user=eHrLyCMAAAAJ",
"https://scholar.google.com/citations?hl=en&user=5tLMJ9sAAAAJ",
"https://scholar.google.com/citations?hl=en&user=phtnuK8AAAAJ",
"https://scholar.google.com/citations?hl=en&user=A-s9GzQAAAAJ",
"https://scholar.google.com/citations?hl=en&user=MGhwUCEAAAAJ",
"https://scholar.google.com/citations?hl=en&user=_ZI3r7wAAAAJ",
"https://scholar.google.com/citations?hl=en&user=RuBwB8wAAAAJ",
"https://scholar.google.com/citations?hl=en&user=HxtIYdwAAAAJ"


]

# Function to extract user ID from URL
def extract_user_id(url):
    return parse_qs(urlparse(url).query).get("user", [""])[0]

# Function to get researcher data from Scholarly
def get_researcher_data(user_id):
    try:
        search_query = scholarly.search_author_id(user_id)
        author = scholarly.fill(search_query)
        return {
            "RName": author.get("name"),
            "Affiliation": author.get("affiliation"),
            "Interests": ", ".join(author.get("interests", [])),
            "Email": author.get("email_domain"),  # May be None
            "Total_Citations": author.get("citedby", 0),
            "Total_Publications": len(author.get("publications", [])),
            "URL": f"https://scholar.google.com/citations?user={user_id}",
            "H_Index": author.get("hindex", 0)
        }
    except Exception as e:
        print(f"[!] Failed to fetch data for {user_id}: {e}")
        return None

# MySQL DB connection
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="@26March2002",
        database="scholars_data"
    )

# Insert researcher into DB
def insert_researcher(cursor, data):
    query = """
        INSERT INTO RESEARCHERS 
        (RName, Affiliation, Interests, Email, Total_Citations, Total_Publications, URL, H_Index)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE 
            RName=VALUES(RName),
            Affiliation=VALUES(Affiliation),
            Interests=VALUES(Interests),
            Total_Citations=VALUES(Total_Citations),
            Total_Publications=VALUES(Total_Publications),
            H_Index=VALUES(H_Index);
    """
    values = (
        data["RName"], data["Affiliation"], data["Interests"], data["Email"],
        data["Total_Citations"], data["Total_Publications"], data["URL"], data["H_Index"]
    )
    cursor.execute(query, values)

def main():
    conn = connect_db()
    cursor = conn.cursor()
    
    for url in urls:
        user_id = extract_user_id(url)
        if not user_id:
            print(f"[!] Invalid URL: {url}")
            continue

        print(f"Fetching data for user ID: {user_id}")
        data = get_researcher_data(user_id)
        if data:
            insert_researcher(cursor, data)
            conn.commit()
            print(f"[+] Inserted: {data['RName']}")
        time.sleep(5)  # polite delay to avoid being rate-limited

    cursor.close()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    main()

def export_researchers_to_csv():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="@26March2002",  # Replace with your MySQL password
        database="scholars_data"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM RESEARCHERS")

    with open("researchers.csv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([i[0] for i in cursor.description])  # Write headers
        writer.writerows(cursor.fetchall())  # Write data

    conn.close()
    print("✅ Exported RESEARCHERS to researchers.csv")

export_researchers_to_csv()