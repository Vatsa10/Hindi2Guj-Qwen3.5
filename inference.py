from train import translate

def simple_infer():
    result = translate("राम बाजार से सब्जियां खरीदकर घर वापस आया।")
    print(result)

def openai_way_infer():
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="outputs/hin-guj/final",
        messages=[
            {
                "role": "user",
                "content": (
                    "Translate the following Hindi sentence to Gujarati.\n\n"
                    "Hindi: माँ, चलो कल एक फिल्म देखने चलते हैं।\n"
                    "Gujarati:"
                )
            }
        ],
        max_tokens=200,
        temperature=0.0,
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    simple_infer()
    # openai_way_infer()