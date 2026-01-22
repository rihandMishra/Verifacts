import sys
from verifacts_pipeline import check_health_claim

EXAMPLE_QUERIES = [
    "Doctors won’t tell you this but vaccines cause autism",
    "Turmeric cures diabetes",
    "पोलियो की दवा बच्चों के लिए खतरनाक है",
]


def run_example_demo():
    print("\n==============================")
    print(" VeriFacts Health – Instant Demo")
    print("==============================\n")

    for query in EXAMPLE_QUERIES:
        print(f"> {query}\n")

        results = check_health_claim(query)

        for res in results:
            print(f"Claim: {res['claim']}\n")
            print(res["response"])
            print("\n----------------------\n")

    print("End of demo.\n")


def interactive_cli():
    print("\n==============================")
    print(" VeriFacts Health CLI")
    print(" Health Misinformation Checker")
    print("==============================\n")

    print("Type a health-related claim.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("> ").strip()

        if user_input.lower() in {"exit", "quit"}:
            print("\nExiting VeriFacts. Stay informed! 👋")
            break

        if not user_input:
            print("Please enter a valid claim.\n")
            continue

        results = check_health_claim(user_input)

        print("\n--- Analysis Result ---\n")

        for res in results:
            print(f"Claim: {res['claim']}\n")
            print(res["response"])
            print("\n----------------------\n")


if __name__ == "__main__":
    if "--example" in sys.argv:
        run_example_demo()
    else:
        interactive_cli()
