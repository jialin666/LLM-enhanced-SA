import pandas as pd
from lifelines import KaplanMeierFitter
import json
import re

from GBSG_generate_prompts_3 import generate_target_prompts, render_features_block, fill_template, generate_narrative
from GBSG_score_function import score

NUM_TEMPLATES = 50
NUMBER_OF_TEST_SAMPLES = 50

# Step1:data preparation:
# reand 1_gbsg_all.csv
_df_train = pd.read_csv("data/1_gbsg_train.csv")
_df_test = pd.read_csv("data/1_gbsg_test.csv")
# df is the combined of _df_train and _df_test
df = pd.concat([_df_train, _df_test])
# if event_occured is true, then the value of time is the event_time or the observation_time
# df['time'] = df['event_occurred'] * df['event_time'] + (1 - df['event_occurred']) * df['observation_time']
# just choose 10 rows randomly
df_test = df.sample(NUMBER_OF_TEST_SAMPLES).copy()

# data description:
DATASET_DESCRIPTION = """Source: https://www.kaggle.com/datasets/utkarshx27/breast-cancer-dataset-used-royston-and-altman?resource=download
Description:
    The data set contains patient records from a 1984-1989 trial conducted by the German Breast Cancer Study Group (GBSG) of 720 patients with node positive breast cancer; 
    it retains the 686 patients with complete data for the prognostic variables.
    These data sets are used in the paper by Royston and Altman(2013). 
   """
    # feature list:
FEATURE_LIST = """
    age: age, years
    meno: menopausal status (0= premenopausal, 1= postmenopausal)
    size: tumor size, mm
    grade: tumor grade
    nodes: number of positive lymph nodes
    pgr: progesterone receptors (fmol/l)
    er: estrogen receptors (fmol/l)
    hormon: hormonal therapy, 0= no, 1= yes
    """
    # rfstime:	recurrence-free survival time; days to first recurrence, death, or last follow-up
    # status:	0= alive without recurrence, 1= recurrence or death

# Step2: generate templates:
# 1) Build the meta prompt and get templates
# meta_prompt = build_meta_prompt(dataset_description, feature_list)
# templates = generate_templates_via_openai(meta_prompt, count=5, model="gpt-4o")
# generate_target_prompts can generate 5 templates each time, but I need to generate NUM_TEMPLATES templates

all_templates = generate_target_prompts(DATASET_DESCRIPTION, FEATURE_LIST, n=NUM_TEMPLATES, model="gpt-4o")
    
# Filter templates: keep only those with correct FEATURES placeholder
valid_templates = {}
key = 0
for template in all_templates:
    template_text = template["template"]
    
    # First, normalize {{FEATURES}} to {FEATURES}
    if "{{FEATURES}}" in template_text:
        template_text = template_text.replace("{{FEATURES}}", "{FEATURES}")
        template["template"] = template_text
    
    # Check if template contains ONLY {FEATURES} placeholder (no other placeholders)
    # Find all placeholders in the format {something}
    placeholders = re.findall(r'\{[^}]+\}', template_text)
    dictator = True
    for placeholder in placeholders:
        if placeholder != "{FEATURES}":
            dictator = False
            break
    if dictator:
        valid_templates[key] = template
        key += 1
    else:
        print(f"Removing template with invalid placeholders: {placeholders}")

# templates = generate_target_prompts(dataset_description, feature_list, n=NUM_TEMPLATES, model="gpt-4o")
with open("1_gbsg_target_prompts.json", "w", encoding="utf-8") as f:
    json.dump(valid_templates, f, indent=2, ensure_ascii=False)
print(f"Saved {len(all_templates)} templates to target_prompts.json")
print("1: templates generated")
for template in valid_templates.values():
    print(template["template"])
    print("="*100)
print("="*100)

# Step3: compute the category of the data:
# df should have columns: 'time', 'event' (1=event, 0=censored)
kmf = KaplanMeierFitter()
kmf.fit(df["time"], event_observed=df["event"])
# Compute survival prob for each patient at their observed time
df["surv_prob"] = kmf.survival_function_at_times(df["time"]).values
# Compute quantiles
low_cut = df["surv_prob"].quantile(0.33)
high_cut = df["surv_prob"].quantile(0.66)
def categorize(p):
    if p <= low_cut: return "low"
    elif p <= high_cut: return "intermediate"
    return "high"
df["category"] = df["surv_prob"].apply(categorize)

# Also compute category for df_test
df_test["surv_prob"] = kmf.survival_function_at_times(df_test["time"]).values
df_test["category"] = df_test["surv_prob"].apply(categorize)

# Step4: iterate over the templates and data:
templates_grades = {} # key: template_id, value: how many times the template predict correctly. 
template_2_id = {}
id_2_template = {}
# fill template_2_id and id_2_template:
for key, template in valid_templates.items():
    template_2_id[template["template"]] = key
    id_2_template[key] = template["template"]
# test template one by one:
print("2: print the filled prompt")
for template_id in id_2_template:
    template = id_2_template[template_id]
    for index, row in df_test.iterrows():
        sample = {
            "age": row['age'],
            "meno": row['meno'],
            "grade": row['grade'],
            "hormon": row['hormon'],
            "size": row['size'],
            "nodes": row['nodes'],
            "pgr": row['pgr'],
            "er": row['er'],
        }
        # Choose a display style for the FEATURES block
        features_block = render_features_block(sample, fmt="bullet")  # bullet/table/json/narrative
        # fill the template with the features
        filled_prompt = fill_template(template, features_block)
        print("?"*100)
        print(filled_prompt)
        # generate the report
        report_text = generate_narrative(filled_prompt, model="gpt-4o")
        print("@"*100)
        print(report_text)
        print("@"*100)

        # get the grade of the report
        grade = score(report_text, row['category'])
        if template_id not in templates_grades:
            templates_grades[template_id] = 0
        templates_grades[template_id] += grade

print(templates_grades)
# save templates_grades to a json file
with open("1_gbsg_templates_grades.json", "w", encoding="utf-8") as f:
    json.dump(templates_grades, f, indent=2, ensure_ascii=False)

# print all templates and their grades
for template_id in templates_grades:
    print("The template is:", id_2_template[template_id])
    print("The grade is:", templates_grades[template_id])
    print("="*100)


