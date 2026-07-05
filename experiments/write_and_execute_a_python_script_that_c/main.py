import anthropic


def get_copenhagen_tattoo_recommendations() -> str:
    """
    Calls Claude Haiku API to generate recommendations for the best tattoo shops
    in Copenhagen with details about each shop.
    
    Returns:
        str: The formatted recommendations text
    """
    client: anthropic.Anthropic = anthropic.Anthropic()
    
    message: anthropic.types.Message = client.messages.create(
        model="claude-haiku-20240307",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": (
                    "List the best tattoo shops in Copenhagen, Denmark. "
                    "For each shop, provide: "
                    "1. Shop name "
                    "2. Neighborhood/location "
                    "3. Style specialty (e.g., traditional, abstract, realism, etc.) "
                    "4. A brief explanation of why it is recommended. "
                    "Format each entry clearly with these labeled sections."
                ),
            }
        ],
    )
    
    recommendations: str = message.content[0].text
    
    print("=" * 60)
    print("BEST TATTOO SHOPS IN COPENHAGEN")
    print("=" * 60)
    print()
    print(recommendations)
    print()
    print("=" * 60)
    
    return recommendations


if __name__ == "__main__":
    get_copenhagen_tattoo_recommendations()