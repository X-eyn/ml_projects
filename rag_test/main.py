import asyncio
from enhanced_search import EnhancedBookSearch


async def main():
    try:

        search_engine = EnhancedBookSearch()

        print("\nStarting book processing...")
        await search_engine.process_book("C:/Users/Admin/Downloads/goodfellow2020.pdf")

        print("\nPerforming search...")
        query = """What is random latent variable?"""

        results = await search_engine.semantic_search(query, k=5)

        for i, result in enumerate(results, 1):
            print(f"\n{'='*50}")
            print(f"Result {i} (Score: {result['relevance_score']:.4f})")
            print(f"Page: {result['metadata'].get('page', 'Unknown')}")
            print(f"{'='*50}")
            print(f"{result['content']}\n")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":

    asyncio.run(main())


#Run 