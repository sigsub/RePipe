from openai import OpenAI
from loading_paper_utils import load_data_from_pkl
from tqdm import tqdm
tqdm.pandas()
import time
import openai
# Initialize the OpenAI client (make sure your API key is set in your environment or here)
client = OpenAI(
    api_key="YOUR API KEY"  # Uncomment and set if not using environment variable
)






def generate_chat_completion(
    user_prompt,
    system_prompt="Summarize the user's input article. The summary must not exceed 150 words exactly.",
    model="gpt-4.1",
    max_tokens=512,
):
    """
    Generate a response using OpenAI chat completion models (e.g., gpt-4.1).
    
    Args:
        user_prompt (str): The user's input prompt.
        system_prompt (str): Instructions to guide the assistant's behavior.
        model (str): The chat model to use (e.g., 'gpt-4.1').
        max_tokens (int): Maximum number of tokens in the output.
        temperature (float): Sampling temperature.
    
    Returns:
        str: The generated assistant response.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": system_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt}
            ]
        }
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "text"},
        temperature=1,
        top_p=1.0,
        max_completion_tokens=max_tokens
    )
    out = response.choices[0].message.content
    if len(out) == 0:
        raise ValueError("The response from the model is empty. Please check the input or model configuration.")
    # wait half a second to avoid hitting rate limits
    time.sleep(0.5)
    return out

def main():
    # example usage
    test_set = load_data_from_pkl('YOUR/TEST/SET/PATH.pkl')
    summaries = []
    for x in tqdm(test_set['pars']):
        try:
            summary = generate_chat_completion(user_prompt=' '.join(x)[:100000])
        except openai.RateLimitError:
            print("Rate limited. Waiting 15 seconds...")
            time.sleep(15)
            summary = generate_chat_completion(user_prompt=' '.join(x)[:100000])
        summaries.append(summary)

    test_set['LLM_summ'] = summaries
    test_set.to_pickle('UPDATED/TEST/SET/WITH/SUMMARIES.pkl')

if __name__ == "__main__":
    main()