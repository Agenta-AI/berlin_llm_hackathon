import os

import agenta as ag
import numpy as np
import openai
import pandas as pd
import weaviate


def get_embedding(text):
    # Use the same embedding generator as what was used on the data!!!
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def summarize(chat_text, summarize_prompt):
    # Summarize conversations since individually they are long and go over 8k limit
    prompt = "Summarize the following conversation on the MLOps.community slack channel. Do not use the usernames in the summary. ```" + chat_text + "```"
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content


def extract_answer(chat_texts, question, summarize_prompt: str, generate_prompt: str):
    # Combine the summaries into a prompt and use SotA GPT-4 to answer.
    prompt = generate_prompt
    for i, chat_text in enumerate(chat_texts):
        prompt += f"\nConversation {i+1} Summary:\n```\n{summarize(chat_text, summarize_prompt)}```"

    if not question.endswith("?"):
        question = question + "?"
    prompt += f"\nQuestion: {question}"
    completion = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    content = completion.choices[0].message.content
    return content


def search_index(client, embedding):
    custom_vector = np.array(embedding).astype(np.float32)
    response = (
        client.query
        .get("Conversations", ["thread_id"])
        .with_near_vector({"vector": custom_vector})
        .with_limit(3)
        .do()
    )

    return response["data"]["Get"]["Conversations"]


def setup():
    openai.organization = os.environ['OPENAI_ORGANIZATION']
    openai.api_key = os.environ['OPENAI_API_KEY']


def read_chats() -> pd.DataFrame:
    chats_df = pd.read_csv("./chats.csv")
    chats_index = {}
    for _, row in chats_df.iterrows():
        chats_index[row['thread_id']] = row['chat_text']
    return chats_index


default_summarize_prompt = "Summarize the following conversation on the MLOps.community slack channel. Do not use the usernames in the summary. ```"
default_generate_prompt = "Use the following summaries of conversations on the MLOps.community slack channel backtics to generate an answer for the user question."


@ag.post
def get_answer(question: str, summarize_prompt: ag.TextParam = default_summarize_prompt, generate_prompt: ag.TextParam = default_generate_prompt) -> str:
    # Get answer to the question by finding the three conversations that are nearest
    # to the question and then using them to generate the answer.
    # Not your Org Name. Org Id can be found in your organization's settings page.
    setup()
    client = weaviate.Client(os.environ['WEVIATE_URL'])
    # Searching documents nearest to the question
    search_vector = get_embedding(question)
    docs = search_index(client, search_vector)
    # Take the top three answers, and use ChatGPT to form the answer to give the user.
    chats_index = read_chats()
    chat_texts = []
    for doc in docs:
        chat_text = chats_index[doc["thread_id"]]
        chat_texts.append(chat_text)
    if len(chat_texts) > 3:
        chat_texts[:3]
    return extract_answer(chat_texts, question, summarize_prompt, generate_prompt)
