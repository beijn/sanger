# %% 
import json, pandas as pd, seaborn as sns, matplotlib.pyplot as pyplot

sf = pd.read_csv('review-sf.csv', sep='\t', index_col=0)
m = json.load(open('bad-random-0.05.json'))

sf24_inacc = list(sf.query('accurate != "x"').index)
sf24_all = set(sf.index)


sf24_inacc = [m['i2pi'][str(i)] for i in sf24_inacc]
sf24_all = [m['i2pi'][str(i)] for i in sf24_all]

# %%

part = '5%' # '5%'

data = pd.DataFrame(columns=['pi', 'Model Loss', 'sf Review'],
                     data=[[int(pi), L, 'inaccurate' if (pi in sf24_inacc) else 'accurate' if (pi in sf24_all) else 'unlabeled'] for pi, L in enumerate(m['pi2L']) if part=='all' or pi in sf24_all]
                     ).sort_values('Model Loss', ascending=False).reset_index(drop=True)

data # type: ignore

# %% 

fig, ax = pyplot.subplots(figsize=(15,5))
g = sns.barplot(data=data, x=data.index, y='Model Loss', hue='sf Review', palette={'inaccurate':'tab:red', 'accurate':'tab:blue', 'unlabeled':'tab:grey'}, gap=0, dodge=False)
g.set_title('Model and Human Assessment of Annotation Quality')
g.set(xticklabels=[])
g.tick_params(bottom=False)  # remove the tick
g.set(xlabel='Annotation')

## add a vertical line at the 5% mark
cut = (0.5 if part == '5%' else 0.05)*len(data)
g.axvline(cut, color='black', linestyle='--')
ax.text(cut +.5, 107, 'Model', ha='left')
ax.text(cut -.5, 107, 'Review', ha='right')
ax.text(cut + 1, 100, 'inaccurate', ha='left')
ax.text(cut - 1, 100, 'accurate', ha='right')

