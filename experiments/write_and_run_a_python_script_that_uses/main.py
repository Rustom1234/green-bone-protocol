import os
import anthropic


def get_copenhagen_tattoo_recommendations() -> str:
    """
    Fetch 5 Copenhagen tattoo shop recommendations using Claude Haiku.
    
    Returns:
        str: Formatted recommendations with shop name, neighborhood, style specialty, and explanation.
    """
    api_key: str = os.environ["ANTHROPIC_API_KEY"]
    client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)
    
    prompt: str = (
        "List 5 tattoo shops in Copenhagen, Denmark. "
        "For each shop provide: Shop Name, Neighborhood, Style Specialty, "
        "and one sentence explaining why someone should go there. "
        "Format each entry clearly and consistently."
    )
    
    response: anthropic.types.Message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    
    result_text: str = response.content[0].text
    return result_text


if __name__ == "__main__":
    recommendations: str = get_copenhagen_tattoo_recommendations()
    print(recommendations)