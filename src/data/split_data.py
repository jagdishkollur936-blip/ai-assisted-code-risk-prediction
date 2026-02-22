from sklearn.model_selection import train_test_split
from src.config import TARGET_COLUMN,TEST_SIZE,RANDOM_STATE


def split_dataset(df):
    """      
    split dataset into train test split

    parameters:
         df(pd.dataframe):Input Dataset

    Retturn:
         x_train,x_test,y_train,y_test
    
    """
    # seperate features and target
    x=df.drop(TARGET_COLUMN,axis=1)
    y=df[TARGET_COLUMN]
    
    # perform train test split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)
    return x_train,x_test,y_train,y_test