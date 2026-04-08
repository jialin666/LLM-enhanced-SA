import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import os
import json
from GBSG_generate_prompts_3 import render_features_block, fill_template, generate_narrative

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# read data
data_train = pd.read_csv("data/1_gbsg_train.csv") #original data
data_test = pd.read_csv("data/1_gbsg_test.csv") #original data

data_train['generated_texts'] = 'Not generated yet' #new column
data_train['embeddings'] = None #new column
data_test['generated_texts'] = 'Not generated yet' #new column
data_test['embeddings'] = None #new column

# read 1_gbsg_target_prompts.json and 1_gbsg_templates_grades.json
with open("1_gbsg_target_prompts.json", "r", encoding="utf-8") as f:
    target_prompts = json.load(f)
with open("1_gbsg_templates_grades.json", "r", encoding="utf-8") as f:
    templates_grades = json.load(f)
# find key of the maximum grade 
max_grade_key = max(templates_grades, key=templates_grades.get)
# get the template of the maximum grade
max_grade_template = target_prompts[max_grade_key]
max_grade_template = max_grade_template["template"]

# define prompt template
def make_prompt(row, template):
    sample = {
        "age": row['age'],
        "meno": row['meno'],
        "size": row['size'],
        "grade": row['grade'],
        "nodes": row['nodes'],
        "pgr": row['pgr'],
        "er": row['er'],
        "hormon": row['hormon'],
    }
    # Choose a display style for the FEATURES block
    features_block = render_features_block(sample, fmt="bullet")  # bullet/table/json/narrative
    # fill the template with the features
    filled_prompt = fill_template(template, features_block)
    # generate the report
    report_text = generate_narrative(filled_prompt, model="gpt-4o")
    return report_text
generated_texts = []
for i, row in tqdm(data_train.iterrows(), total=len(data_train)):
    report_text = make_prompt(row, max_grade_template)
    generated_texts.append(report_text)
embeddings = []
for i, text in enumerate(generated_texts):
    # print(i)
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        # each embedding is a 1536-dimensional vector
        vector = response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        vector = [None] * 1536

    embeddings.append(vector)
# save to CSV file
num_generated = len(generated_texts)
data_train.loc[:, 'generated_texts'] = generated_texts
for i in range(num_generated):
    data_train.at[data_train.index[i], 'embeddings'] = embeddings[i]  # Use .at with the correct index
# save to CSV file
data_train.to_csv("data/1_gbsg_train_new_with_embeddings.csv", index=False)   

generated_texts = []
# iterate through the data_test
for i, row in tqdm(data_test.iterrows(), total=len(data_test)):
    report_text = make_prompt(row, max_grade_template)
    generated_texts.append(report_text)
embeddings = []
for i, text in enumerate(generated_texts):
    # print(i)
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        # each embedding is a 1536-dimensional vector
        vector = response.data[0].embedding
    except Exception as e:
        print(f"Error: {e}")
        vector = [None] * 1536

    embeddings.append(vector)
# save to CSV file
num_generated = len(generated_texts)
data_test.loc[:, 'generated_texts'] = generated_texts
for i in range(num_generated):
    data_test.at[data_test.index[i], 'embeddings'] = embeddings[i]  # Use .at with the correct index
# save to CSV file
data_test.to_csv("data/1_gbsg_test_new_with_embeddings.csv", index=False)   