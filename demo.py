from verifacts_pipeline import check_health_claim

query = "Doctors won’t tell you this but vaccines cause autism"
result = check_health_claim(query)

print("\nUSER QUERY:")
print(query)
print("\nSYSTEM RESPONSE:\n")

for r in result:
    print(r)
    print("=" * 50)
