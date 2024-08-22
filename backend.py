import pandas as pd
import pickle
import category_encoders as ce


def catCheck(dataFrame, target):
    isCategorical = True
    if (str(type(dataFrame[target][0])) == "<class 'numpy.int64'>" or
            str(type(dataFrame[target][0])) == "<class 'numpy.float64'>"):
        isCategorical = False
    elif (str(type(dataFrame[target][0]))) == "<class 'str'>" and 48 <= ord(dataFrame[target][0][0]) <= 57:
        isCategorical = False
        dataFrame[target] = dataFrame[target].astype('float64')
    elif (str(type(dataFrame[target][0]))) != "<class 'str'>":
        dataFrame[target] = dataFrame[target].astype('float64')
    return isCategorical


def encoding(dataFrame):
    df_category = dataFrame.select_dtypes(exclude=['int64', 'float64'])
    encoder = ce.BaseNEncoder(cols=list(df_category.columns), return_df=True, base=2)
    cat_encoded = encoder.fit_transform(df_category)
    save_cat_encoding(encoder)
    X_cat = pd.concat([cat_encoded, dataFrame.select_dtypes(include=['int64', 'float64'])], axis=1)
    return X_cat


def save_cat_encoding(enc):
    output = open('pickleFiles/tempPickleFile.pkl', 'wb')
    pickle.dump(enc, output)
    output.close()


def load_encoder():
    pklFile = open(r'pickleFiles/tempPickleFile.pkl', 'rb')
    retrieved_encoder = pickle.load(pklFile)
    pklFile.close()
    return retrieved_encoder
