

with open(args.src_avs_path) as f:
    golds = [rec['summmary'] for rec in con]
    srcs = [rec['text'] for rec in con]
with open(args.gen_avs_path) as f: ]
preds = f.readlines()
assert len(srcs) == len(golds) == len(preds)
