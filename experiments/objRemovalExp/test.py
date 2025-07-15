import pandas as pd

print("HI")

if __name__ == "__main__":
    # print("Hello")
    # rows = [
    #     {'b': 1, 'a': 2},
    #     {'a': 3, 'c': 4},
    #     {'c': 5, 'b': 6},
    # ]
    # df = pd.DataFrame(rows)
    # print(df)

    cols = {
        "name": ["John", "John", "Alice", "Bob"],
        "favNum": [1, 2, 3, 4],
        "favLetter": ["x", "y", "z", "w"]
    }
    df = pd.DataFrame(cols)
    print(df)
    print(df.loc[1])
    df = df.set_index("name")
    print(df)
    print(df.loc["John"])
