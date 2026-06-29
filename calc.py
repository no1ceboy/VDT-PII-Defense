import json

d = {'total':0, 'success':0, 'cat':{}, 'mod':{}, 'diff':{}}

with open('results/attack_results.jsonl', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line: continue
        x = json.loads(line)
        
        success = int(x.get('attack_success', 0))
        d['total'] += 1
        d['success'] += success
        
        cat = x.get('attack_category', 'unknown')
        if cat not in d['cat']: d['cat'][cat] = [0,0]
        d['cat'][cat][0] += 1
        d['cat'][cat][1] += success
        
        mod = x.get('model', 'unknown')
        if mod not in d['mod']: d['mod'][mod] = [0,0]
        d['mod'][mod][0] += 1
        d['mod'][mod][1] += success
        
        diff = x.get('attack_difficulty', 'unknown')
        if diff not in d['diff']: d['diff'][diff] = [0,0]
        d['diff'][diff][0] += 1
        d['diff'][diff][1] += success

print(f'Overall: {d["success"]}/{d["total"]} = {d["success"]/max(1,d["total"]):.2%}')
print('Categories:')
for k,v in d['cat'].items(): print(f'  {k}: {v[1]}/{v[0]} = {v[1]/v[0]:.2%}')
print('Models:')
for k,v in d['mod'].items(): print(f'  {k}: {v[1]}/{v[0]} = {v[1]/v[0]:.2%}')
print('Difficulties:')
for k,v in d['diff'].items(): print(f'  {k}: {v[1]}/{v[0]} = {v[1]/v[0]:.2%}')
