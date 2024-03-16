import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

import pickle
import streamlit
import plotly.express

def featureEngineering(df,trainNumerical_cols,trainCategorical_cols):
    
    #pre_trainX = df

    numPipeline = Pipeline(steps=[('standard_scaler',StandardScaler())],verbose=True)
    colPipeline = Pipeline(steps=[('onehot_encoder',OneHotEncoder(sparse_output=False))])
    columnTransformer_pipeline = ColumnTransformer(transformers=[('num',numPipeline,trainNumerical_cols),
                                                    ('col',colPipeline,trainCategorical_cols)
                                                    ],
                                                    remainder='passthrough')
    df = columnTransformer_pipeline.fit_transform(df)
    pickle.dump(columnTransformer_pipeline,open('E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\columnTransformer_pipeline.sav','wb'))   
    return df
    

def processData(df):
    df = df.drop('Id',axis=1)
    #Drop features which have high number of Nulls
    df = df.drop(['PoolQC','MiscFeature','Alley','Fence','MasVnrType','FireplaceQu','LotFrontage'],axis=1)
    return df

def create_model():
    df_test = pd.read_csv("E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\test.csv")
    df_train = pd.read_csv("E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\train.csv")

    df_train = processData(df_train)

    X = df_train.drop(columns=['SalePrice']) #Independent variables/Features
    y = df_train['SalePrice'] #Dependent variable/ Target Feature

    trainNumerical_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    X[trainNumerical_cols] = X[trainNumerical_cols].fillna(X[trainNumerical_cols].median())
    trainCategorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    X[trainCategorical_cols] = X[trainCategorical_cols].fillna(X[trainCategorical_cols].mode().iloc[0])

    pre_trainX = featureEngineering(X,trainNumerical_cols,trainCategorical_cols)
    pre_trainY = y

    pcaPipeline = Pipeline(steps=[('pca',PCA(n_components=36))])

    X_train, X_test, y_train, y_test= train_test_split(pre_trainX,pre_trainY,random_state=42,test_size=0.20)
    #X_train = columnTransformer_pipeline.transform(X_train)
    model = Pipeline(steps=[
                               ('pcaPipeline',pcaPipeline),
                               ('rfr',RandomForestRegressor(random_state=42,n_jobs=-1,max_depth= 20, max_features= 20, min_samples_leaf= 5, n_estimators= 50))
                            ])
    
    model.fit(X_train,y_train)

    y_train_pred = model.predict(X_train)
    print(r2_score(y_train,y_train_pred))
    print(y_train_pred)
    #X_test = columnTransformer_pipeline.fit_transform(X_test)
    #print(r2_score(y_test,model.predict(X_test)))

    pickle.dump(model,open('E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\model.sav','wb'))

    return model

create_model()

def load_columnTransformer():
    return pickle.load(open('E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\columnTransformer_pipeline.sav','rb'))
def load_model():
    return pickle.load(open('E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\model.sav','rb'))


def predict(X):
    model = load_model()
    print("test:",model.predict(X))
    return model.predict(X)

if 'predicted_price' not in streamlit.session_state:
    streamlit.session_state['predicted_price'] = '0'

def streamlitUI():
    streamlit.title("House Price Prediction App")

    container_style = """
    <style>
        .container1 {
            border: 2px solid #3498db;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 20px;
        }
    </style>
    """

    container2 = streamlit.container(height=800)
    with container2:
        container2.markdown("<div class='container1'>", unsafe_allow_html=True)
        container2.subheader("Parameters")
        col1,col2 = container2.columns(2)
        with col1:
            msZoning = col1.radio('MSZonning',list(df_test['MSZoning'].unique()),horizontal=True)
        with col2:
            lotArea = col2.select_slider('LotArea',[i for i in range(df_test['LotArea'].min(),df_test['LotArea'].max())])
        container2.markdown("</div>", unsafe_allow_html=True)
        
        col1,col2 = container2.columns(2)
        with col1:
            bldgType = col1.radio('BldgType',list(df_test['BldgType'].unique()),horizontal=True)
        with col2:
            houseStyle = col2.radio('HouseStyle',list(df_test['HouseStyle'].unique()),horizontal=True)

        col1,col2 = container2.columns(2)
        with col1:
            overallQual = col1.select_slider('OverallQual',[i for i in range(df_test['OverallQual'].min(),df_test['OverallQual'].max())])
        with col2:
            overallCond = col2.select_slider('OverallCond',[i for i in range(df_test['OverallCond'].min(),df_test['OverallCond'].max())])

        streamlit.subheader("Bedrooms")
        col1,col2 = streamlit.columns(2)
        with col1:
            BedroomAbvGr = streamlit.select_slider('BedroomAbvGr',[i for i in range(int(df_test['BedroomAbvGr'].min()),int(df_test['BedroomAbvGr'].max()))])
        with col2:
            TotRmsAbvGrd = streamlit.select_slider('TotRmsAbvGrd',[i for i in range(int(df_test['TotRmsAbvGrd'].min()),int(df_test['TotRmsAbvGrd'].max()))])
        

        streamlit.subheader("Bathrooms")

        col1,col2,col3,col4 = streamlit.columns(4)
        with col1:
            BsmtFullBath = col1.radio('BsmtFullBath',list(df_test['BsmtFullBath'].unique()),horizontal=True)
        with col2:
            BsmtHalfBath = col2.radio('BsmtHalfBath',list(df_test['BsmtHalfBath'].unique()),horizontal=True)
        with col3:
            FullBath = col3.radio('FullBath',list(df_test['FullBath'].unique()),horizontal=True)
        with col4:
            HalfBath = col4.radio('HalfBath',list(df_test['HalfBath'].unique()),horizontal=True)    

        streamlit.subheader("Garage")
        colGarage1,colGarage2,colGarage3 = streamlit.columns(3)
        with colGarage1:
            GarageType = colGarage1.radio('GarageType',list(df_test['GarageType'].unique()),horizontal=True)
        with colGarage2:
            GarageCars = colGarage2.select_slider('GarageCars',[i for i in range(int(df_test['GarageCars'].max()))])
        with colGarage3:
            GarageArea = colGarage3.select_slider('GarageArea',[i for i in range(int(df_test['GarageArea'].min()),int(df_test['GarageArea'].max()))])

        streamlit.subheader("Sale Type & Condition")
        col1,col2 = streamlit.columns(2)
        with col1:
            SaleType = col1.radio('SaleType',list(df_test['SaleType'].unique()),horizontal=True)
        with col2:
            SaleCondition = col2.radio('SaleCondition',list(df_test['SaleCondition'].unique()),horizontal=True)
    
    container1 = streamlit.container(height=120)
    with container1:
        container1.markdown("<div class='container1'>", unsafe_allow_html=True)
        container1.subheader("Predicted Price: ")
        container1.text(streamlit.session_state['predicted_price'])
        container1.markdown("</div>", unsafe_allow_html=True)

    df_val = pd.DataFrame({
    "Id":[None],
    "MSSubClass":df_test['MSSubClass'].mode()[0],
    "MSZoning":msZoning,
    "LotFrontage":df_test['LotFrontage'].mode()[0],
    "LotArea":lotArea,
    "Street":df_test['Street'].mode()[0],
    "Alley":df_test['Alley'].mode()[0],
    "LotShape":df_test['LotShape'].mode()[0],
    "LandContour":df_test['LandContour'].mode()[0],
    "Utilities":df_test['Utilities'].mode()[0],
    "LotConfig":df_test['LotConfig'].mode()[0],
    "LandSlope":df_test['LandSlope'].mode()[0],
    "Neighborhood":df_test['Neighborhood'].mode(),
    "Condition1":df_test['Condition1'].mode(),
    "Condition2":df_test['Condition2'].mode(),
    "BldgType":bldgType,
    "HouseStyle":houseStyle,
    "OverallQual":overallQual,
    "OverallCond":overallCond,
    "YearBuilt":df_test['YearBuilt'].mode(),
    "YearRemodAdd":df_test['YearRemodAdd'].mode(),
    "RoofStyle":df_test['RoofStyle'].mode(),
    "RoofMatl":df_test['RoofMatl'].mode(),
    "Exterior1st":df_test['Exterior1st'].mode(),
    "Exterior2nd":df_test['Exterior2nd'].mode(),
    "MasVnrType":df_test['MasVnrType'].mode(),
    "MasVnrArea":df_test['MasVnrArea'].mode(),
    "ExterQual":df_test['ExterQual'].mode(),
    "ExterCond":df_test['ExterCond'].mode(),
    "Foundation":df_test['Foundation'].mode(),
    "BsmtQual":df_test['BsmtQual'].mode()[0],
    "BsmtCond":df_test['BsmtCond'].mode()[0],
    "BsmtExposure":df_test['BsmtExposure'].mode()[0],
    "BsmtFinType1":df_test['BsmtFinType1'].mode()[0],
    "BsmtFinSF1":df_test['BsmtFinSF1'].mode()[0],
    "BsmtFinType2":df_test['BsmtFinType2'].mode()[0],
    "BsmtFinSF2":df_test['BsmtFinSF2'].mode()[0],
    "BsmtUnfSF":df_test['BsmtUnfSF'].mode()[0],
    "TotalBsmtSF":df_test['TotalBsmtSF'].mode()[0],
    "Heating":df_test['Heating'].mode()[0],
    "HeatingQC":df_test['HeatingQC'].mode()[0],
    "CentralAir":df_test['CentralAir'].mode()[0],
    "Electrical":df_test['Electrical'].mode()[0],
    "1stFlrSF":df_test['1stFlrSF'].mode()[0],
    "2ndFlrSF":df_test['2ndFlrSF'].mode()[0],
    "LowQualFinSF":df_test['LowQualFinSF'].mode()[0],
    "GrLivArea":df_test['GrLivArea'].mode()[0],

    "BsmtFullBath":BsmtFullBath,
    "BsmtHalfBath":BsmtHalfBath,
    "FullBath":FullBath,
    "HalfBath":HalfBath,

    "BedroomAbvGr":BedroomAbvGr,
    "KitchenAbvGr":df_test['KitchenAbvGr'].mode()[0],
    "KitchenQual":df_test['KitchenQual'].mode()[0],
    "TotRmsAbvGrd":TotRmsAbvGrd,
    "Functional":df_test['Functional'].mode()[0],
    "Fireplaces":df_test['Fireplaces'].mode()[0],
    "FireplaceQu":df_test['FireplaceQu'].mode()[0],
    "GarageType":GarageType,
    "GarageYrBlt":df_test['GarageYrBlt'].mode()[0],
    "GarageFinish":df_test['GarageFinish'].mode()[0],
    "GarageCars":GarageCars,
    "GarageArea":GarageArea,
    "GarageQual":df_test['GarageQual'].mode()[0],
    "GarageCond":df_test['GarageCond'].mode()[0],
    "PavedDrive":df_test['PavedDrive'].mode()[0],
    "WoodDeckSF":df_test['WoodDeckSF'].mode()[0],
    "OpenPorchSF":df_test['OpenPorchSF'].mode()[0],
    "EnclosedPorch":df_test['EnclosedPorch'].mode()[0],
    "3SsnPorch":df_test['3SsnPorch'].mode()[0],
    "ScreenPorch":df_test['ScreenPorch'].mode()[0],
    "PoolArea":df_test['PoolArea'].mode()[0],
    "PoolQC":df_test['PoolQC'].mode()[0],
    "Fence":df_test['Fence'].mode()[0],
    "MiscFeature":df_test['MiscFeature'].mode()[0],
    "MiscVal":df_test['MiscVal'].mode()[0],
    "MoSold":df_test['MoSold'].mode()[0],
    "YrSold":df_test['YrSold'].mode()[0],
    "SaleType":SaleType,
    "SaleCondition":SaleCondition
    })
    return df_val

df_test = pd.read_csv("E:\\Edureka\\Code\\SelfPython\\Kaggle\\HousePrices\\test.csv")

trainNumerical_cols = df_test.select_dtypes(include=['int64','float64']).columns.tolist()
df_test[trainNumerical_cols] = df_test[trainNumerical_cols].fillna(df_test[trainNumerical_cols].median())
trainCategorical_cols = df_test.select_dtypes(include=['object']).columns.tolist()
df_test[trainCategorical_cols] = df_test[trainCategorical_cols].fillna(df_test[trainCategorical_cols].mode().iloc[0])

df_val = streamlitUI()
#print(df_val)
df_val = processData(df_val)

#df_val = featureEngineering(df_val,trainNumerical_cols,trainCategorical_cols)
columnTransformerModel = load_columnTransformer()
df_val = columnTransformerModel.transform(df_val)
pred = round(predict(df_val)[0])
print("pred:",pred)

streamlit.session_state['predicted_price'] = pred
#fig3 = plotly.express.line(pred)
#fig3.update_layout(
#    streamlit.plotly_chart(fig3)
#)
