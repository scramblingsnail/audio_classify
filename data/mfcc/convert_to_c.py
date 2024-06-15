import pathlib

defs = []
for f in pathlib.Path(__file__).parent.iterdir():
    vardef = ""
    if not f.name.endswith(".txt"):
        continue
    lns = f.read_text().splitlines()
    row_count = len(lns)
    col_count = len(lns[0].split(" "))
    for ln in lns:
        if col_count != 1:
            vardef += "{"
        for v in ln.split(" "):
            vardef += f"{v},"
        if col_count != 1:
            vardef += "},"
    vardef_prefix = f"static const double {f.name[:-4]}[{row_count}]"
    if col_count != 1:
        vardef_prefix += f"[{col_count}]"
    vardef_prefix += "="
    vardef = vardef_prefix + "{" + vardef + "};"
    defs.append(vardef)
pathlib.Path(__file__).parent.joinpath('./lstm_params.c').write_text('\n'.join(defs))