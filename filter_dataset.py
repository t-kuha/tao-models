import json

import polars as pl

# classmap (class names)
with open('classmap.json') as f:
    classmap = json.load(f)

# class names vs. label name
df = pl.read_csv(
    '2017_11/class-descriptions.csv',
    has_header=False,
    new_columns=['id', 'label']
)
# print(df)

# not found:
# Bicycle:
# Football:
# Office building:
# Vehicle registration plate: Licence plate

# extract ids from `df` where the label is present in `classmap`
valid_ids = df.filter(pl.col('label').is_in(list(classmap.keys()))).get_column('id').to_list()

# Label Name
annot = pl.read_csv(
    '2017_11/validation/annotations-human.csv',
    schema_overrides={
        'ImageID': pl.Utf8,
        'Source': pl.Utf8,
        'LabelName': pl.Utf8,
        'Confidence': pl.Int8,
    }
)
# keep only entries whose LabelName is in valid_ids and Confidence == 1
annot = annot.filter(
    pl.col('LabelName').is_in(valid_ids) & (pl.col('Confidence') == 1)
)

# ImageIDs vs. class name
id_vs_cls = annot.join(
    df, left_on='LabelName', right_on='id', how='left').select(
        ['ImageID', 'label']
    ).rename(
        {'label': 'ClassName'}
    )

# count unique ImageIDs
# print(len(id_vs_cls), len(id_vs_cls.select('ImageID').unique()))

# convert to mapping ImageID -> ClassName and save as JSON
records = id_vs_cls.to_dicts()
# keep all ClassName values per ImageID as a list (preserve order, remove duplicates)
mapping = {}
for r in records:
    mapping.setdefault(r['ImageID'], []).append(r['ClassName'])
for k, v in mapping.items():
    mapping[k] = list(dict.fromkeys(v))
with open('id_vs_classname.json', 'w', encoding='utf-8') as f:
    json.dump(mapping, f, indent=2, ensure_ascii=False)
