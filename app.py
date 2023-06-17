import os

import agenta as ag
import numpy as np
import openai
import pandas as pd
import weaviate
import cohere
import replicate


def get_embedding(text):
    # Use the same embedding generator as what was used on the data!!!
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


def summarize(chat_text, summarize_prompt):
    # Summarize conversations since individually they are long and go over 8k limit
    prompt = summarize_prompt + chat_text + "```"

    return ask_replicate(prompt)


def ask_cohere(prompt: str) -> str:
    co = cohere.Client(os.environ['COHERE_API_KEY'])
    response = co.generate(
        prompt=prompt)
    content = response.generations[0].text
    return content


def ask_replicate(prompt: str) -> str:
    output = replicate.run(
        "replicate/vicuna-13b:6282abe6a492de4145d7bb601023762212f9ddbbe78278bd6771c8b3b2f2a13b",
        input={"prompt": prompt}
    )
    result = ""
    for item in output:
        result += item
    return result


def extract_answer(chat_texts, question, summarize_prompt: str, generate_prompt: str):
    # Combine the summaries into a prompt and use SotA GPT-4 to answer.
    prompt = generate_prompt
    for i, chat_text in enumerate(chat_texts):
        prompt += f"\nConversation {i+1} Summary:\n```\n{summarize(chat_text, summarize_prompt)}```"

    if not question.endswith("?"):
        question = question + "?"
    prompt += f"\nQuestion: {question}"

    return ask_replicate(prompt)


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
