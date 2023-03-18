import pandas as pd
import random

def create_dataframe() ->pd.DataFrame:
    classes = [*('A'*800), *('B'*150), *('C'*50)]
    random.shuffle(classes)
    df = pd.DataFrame(
        {
            'value': [random.randint(0, 10) for _ in range(1000)],
            'class': classes
        }
    )
    return df

def manual_split(df: pd.DataFrame) -> tuple:
    size = df.shape[0]
    train = df.iloc[:int(size*0.8)]
    test = df.iloc[int(size*0.8):]
    return train, test

def sklearn_split(df: pd.DataFrame) -> tuple:
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2)
    return train, test

def print_stats(df: pd.DataFrame) -> None:
    print(df.shape)
    print(df['class'].value_counts()/df.shape[0])


def main():
    df = create_dataframe()
    train1, test1 = manual_split(df)
    print_stats(train1)
    print_stats(test1)
    train2, test2 = sklearn_split(df)
    print_stats(train2)
    print_stats(test2)

if __name__ == '__main__':
    main()