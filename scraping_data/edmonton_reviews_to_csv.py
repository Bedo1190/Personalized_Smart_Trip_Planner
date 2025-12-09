import csv
import json

CSV_FILE = "edmonton_businesses.csv"       # içinde business_id olan dosya
REVIEW_JSON = "review.json"                    # Yelp review dataset
OUTPUT_FILE = "matched_reviews.csv"            # eşleşen yorumların çıkacağı dosya

# 1) CSV'den business_id'leri oku
valid_business_ids = set()

with open(CSV_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        bid = row.get("business_id")
        if bid:
            valid_business_ids.add(bid)

print(f"{len(valid_business_ids)} tane business_id yüklendi!")

# 2) Eşleşen yorumları çek
with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    writer.writerow(["business_id", "review_id", "stars", "text", "date"])

    with open(REVIEW_JSON, encoding="utf-8") as r_f:
        for line in r_f:
            review = json.loads(line)

            if review["business_id"] in valid_business_ids:
                writer.writerow([
                    review["business_id"],
                    review["review_id"],
                    review["stars"],
                    review["text"],
                    review["date"]
                ])

print("✔ Eşleşen yorumlar kaydedildi:", OUTPUT_FILE)
