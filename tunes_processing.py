def preprocess_tunes(abc):
    abc = abc.strip()
    abc = abc.split("\n")
    for i, line in enumerate(abc):
        if (
            line.startswith("%")
            or line.startswith("X:")
            or line.startswith("T:")
            or line.startswith("S:")
        ):
            abc.pop(i)
        else:
            line = line.replace("\\", "")
    abc.pop(0)
    abc.pop(0)
    abc = "\n".join(abc)
    return abc
