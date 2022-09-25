import pandas as pd

if __name__ == '__main__':

    df = pd.DataFrame(
        {
            "path": [
                r"C:\Users\dkoro\PythonProjects\SuperResolution\modeling\data\images\0001.png",
                r"C:\Users\dkoro\PythonProjects\SuperResolution\modeling\data\images\0002.png",
                r"C:\Users\dkoro\PythonProjects\SuperResolution\modeling\data\images\0001.png",
                r"C:\Users\dkoro\PythonProjects\SuperResolution\modeling\data\images\0002.png",
            ],
            "split": ['train', 'train', 'valid', 'valid']
        }
    )

    df.to_csv(r"C:\Users\dkoro\PythonProjects\SuperResolution\modeling\data\data.csv", index=False)
