import json
import csv

JSON_FILE = "all_businesses.json"
OUTPUT_CSV = "edmonton_businesses.csv"

CATEGORIES = [
    "restaurants",
    "cafes",
    "bars",
    "nightlife",
    "museums",
    "landmarks",
    "hotels",
    "shopping",
    "arts",
    "parks"
]

saved_count = 0

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as out_f:
    writer = csv.writer(out_f)
    writer.writerow([
        "business_id", "name", "rating", "review_count", "category",
        "lat", "lon", "address", "phone", "category_source"
    ])

    with open(JSON_FILE, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            business = json.loads(line)

            if business.get("city", "").lower() != "edmonton":
                continue

            business_categories = business.get("categories") or ""
            business_categories = business_categories.lower()
            matched_category = None
            for cat in CATEGORIES:
                if cat in business_categories:
                    matched_category = cat
                    break

            if not matched_category:
                continue

            phone = ""
            if business.get("attributes") and isinstance(business["attributes"], dict):
                phone = business["attributes"].get("display_phone", "")

            writer.writerow([
                business.get("business_id", ""),
                business.get("name", ""),
                business.get("stars", ""),
                business.get("review_count", ""),
                business.get("categories", ""),
                business.get("latitude", ""),
                business.get("longitude", ""),
                business.get("address", ""),
                phone,
                matched_category
            ])
            saved_count += 1

print(f"✔ Toplam kaydedilen işletme sayısı: {saved_count}")
print(f"✔ CSV kaydedildi")
