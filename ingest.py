import json
import os
from typing import List

import agenta as ag
import numpy as np
import pandas as pd
import weaviate


def load_data() -> List:
    message_embeddings_df = pd.read_csv("./Vector DB & LLM Hackathon/messages-embeddings-ada-002.csv")
    chats_df = pd.read_csv("./Vector DB & LLM Hackathon/chats.csv")
    embeddings_df = pd.read_csv("./Vector DB & LLM Hackathon/chats-embeddings-ada-002.csv")
    # Create a temp index of the chats
    chats_index = {}
    for _, row in chats_df.iterrows():
        chats_index[row['thread_id']] = row['chat_text']

    # Link the chats and embeddings together
    embeddings = []
    VECTOR_SIZE = None
    for _, row in embeddings_df.iterrows():
        embedding = json.loads(row['embedding'])
        embeddings.append({"thread_id": row['thread_id'], "embedding":  embedding})
        if not VECTOR_SIZE:
            VECTOR_SIZE = len(embedding)
        else:
            assert VECTOR_SIZE == len(embedding)
    return embeddings


def add_data(embeddings: List):
    class_name = "Conversations2"
    client = weaviate.Client(os.environ['WEVIATE_URL'])
    client.schema.get()
    # Create a class object for our chat conversations:
    class_obj = {
        "class": class_name,
        "vectorizer": "none",  # None is custom vectorizer
    }

    client.schema.create_class(class_obj)
    # bulk insert data
    with client.batch(
        batch_size=100
    ) as batch:
        # Batch import all conversations
        for row in embeddings:
            properties = {
                "thread_id": row['thread_id'],
            }

            custom_vector = np.array(row['embedding']).astype(np.float32)
            client.batch.add_data_object(
                properties,
                class_name,
                vector=custom_vector  # Add custom vector
            )


@ag.ingest
def ingest():
    embeddings = load_data()
    add_data(embeddings)
    # make sure schema is empty
