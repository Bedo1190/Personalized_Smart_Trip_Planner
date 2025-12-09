import requests
import csv
import time

API_KEY = ""

headers = {
    "Authorization": f"Bearer {API_KEY}",
}

url = "https://api.yelp.com/v3/businesses/search"

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

csv_file = "paris_1000_mixed_places.csv"

with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)

    writer.writerow([
        "business_id", "name", "rating", "review_count", "category",
        "lat", "lon", "address", "phone", "category_source"
    ])

    for cat in CATEGORIES:
        print(f"Kategori çekiliyor: {cat}")

        limit = 50
        offsets = [0, 50] 

        for offset in offsets:
            params = {
                "location": "Paris",
                "limit": limit,
                "offset": offset,
                "term": cat,
                "sort_by": "rating"
            }

            response = requests.get(url, headers=headers, params=params)
            data = response.json()

            businesses = data.get("businesses", [])

            for b in businesses:
                business_id = b.get("id", "")
                name = b.get("name", "")
                rating = b.get("rating", "")
                review_count = b.get("review_count", "")
                categories = ", ".join([c["title"] for c in b.get("categories", [])])
                lat = b.get("coordinates", {}).get("latitude", "")
                lon = b.get("coordinates", {}).get("longitude", "")
                address = " ".join(b.get("location", {}).get("display_address", []))
                phone = b.get("display_phone", "")

                writer.writerow([
                    business_id, name, rating, review_count, categories,
                    lat, lon, address, phone, cat
                ])

            time.sleep(0.4)

print(f"✔ 1000 mekan + business_id kaydedildi: {csv_file}")
