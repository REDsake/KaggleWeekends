# importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# importing the dataset

data = pd.read_csv('train.csv')
x= data.iloc[ :,1 : -1 ].values
y= data.iloc[: , -1 :].values

# data preprocessing

# taking log of the dataset in order to make it linear

target = np.log( data.SalePrice)
print('Skewness of the target price array is ' , target.skew())
plt.hist( target , color = 'green')

# working with the numeric types only 

numeric_features = data.select_dtypes( include = [np.number] ) 
target = numeric_features['SalePrice']

# finding the correlation bw the numeric_features and the target

corr = numeric_features.corr()
corr = corr['SalePrice']
corr = corr.sort_values( ascending = False )
overallqual = numeric_features['OverallQual'].unique()

# visualising the relationship bw overallQual and target

quality_pivot = data.pivot_table( index = 'OverallQual' , values = 'SalePrice' , aggfunc = np.median )
quality_pivot.plot( kind = 'bar' , color = 'blue' )

# visualising the relation ship bw GrLivArea and Sale price 

quality_pivot2 = data.pivot_table( index = 'GrLivArea' , values = 'SalePrice' , aggfunc = np.median )
plt.scatter( data['SalePrice'] , data['GrLivArea'] )
plt.xlabel('GrLivArea' )
plt.ylabel('Sale price')

# visuailsing the realationship bw the GarageCars

plt.scatter( data['GarageCars'] , data['SalePrice'])

# visulasing the realtionship bw the garage area and the sale price

plt.scatter( data['GarageArea'] , data['SalePrice'])
plt.scatter( data['MoSold'], data['SalePrice'])
plt.hist( data['BedroomAbvGr'] )


# handaling the outliers in the dataset

#data = data[ data['GarageArea'] < 1200 ]
#plt.scatter( data['GarageArea'] , data['SalePrice'])
#data = data[ data['LotFrontage'] < 200 ]
#data = data[ data['BsmtFinSF2'] < 1200 ]
#data = data[ data['TotalBsmtSF']< 3000 ]




################### handaling the missing values ######################

nulls = pd.DataFrame( data.isnull().sum().sort_values( ascending = False )[ : 25] )
nulls.index.name = 'Features'
nulls.columns = [ 'Null Count' ]
misc = data['MiscFeature'].unique()

# for numerical values we are imputing the missing values with mean values

from sklearn.preprocessing import Imputer, StandardScaler
imputer = Imputer( missing_values = "NaN" , strategy = "mean" )
numeric_features = imputer.fit_transform( numeric_features )

sc_x = StandardScaler()
sc_y = StandardScaler()
numeric_features = sc_x.fit_transform( numeric_features )
y = sc_y.fit_transform( y )

# handaling the categorical values

categoricals = data.select_dtypes( exclude = [np.number] )
desc = categoricals.describe()


# encoding the categorical data
d1 = pd.get_dummies( categoricals.MSZoning , drop_first = True , dummy_na = True )
for i in range( 1 , 43 ):
    d1 = np.append( d1 , pd.get_dummies( categoricals.iloc[ : , i ].values , dummy_na = True , drop_first = False ) , axis = 1 )

# final dataset after cleaning 
    
clean_data = np.append( d1 , numeric_features , axis = 1 )

# fitting the decison tree model to the dataset

from sklearn.svm import SVR
regressor = SVR( kernel = 'rbf' )
regressor.fit( clean_data , y )

############ applying the same data preprocessing to the test set ###########

test = pd.read_csv( 'test.csv')
test_categoricals = test.select_dtypes( exclude = [ np.number ] )
test_numericals = test.select_dtypes( include = [ np.number ] )

# excluding the ID index from the arry
test_numericals = test_numericals.iloc[: , 1: ].values

# taking care of missing numerical value by replacing it with average

imputer2 = Imputer( )
test_numericals = imputer2.fit_transform( test_numericals )

test_numericals = sc_x.fit_transform( test_numericals )

# making dummy variabels for categorical data

d2 = pd.get_dummies( test_categoricals.iloc[ : , 0 ].values ,dummy_na = True , drop_first = True )
for i in range( 1, 43 ):
    d2 = np.append( d2 , pd.get_dummies( test_categoricals.iloc[: , i ].values , dummy_na = True , drop_first = True  ) ,axis = 1 )
    
# creating a clean test set4
clean_test_set = np.append( d2 , test_numericals , axis = 1 )


# now making the shape of clean_test_set and clean_data same 

clean_test_set = np.append( clean_test_set , np.zeros( shape = ( 1 , np.shape( clean_test_set)[1] ) ) , axis = 0 )
clean_test_set = np.append( clean_test_set ,np.zeros(shape = (np.shape(clean_data)[0] , np.shape( clean_data )[1] - np.shape( clean_test_set)[1]) ,
                                    dtype = np.int64 ) ,axis = 1 )



######################### predicting the test results #############################

y_pred = pd.DataFrame( )
y_pred['Id'] = test.Id
predictions = regressor.predict( clean_test_set )
predictions = sc_y.inverse_transform( predictions )
predictions = predictions[:-1 ]
y_pred['SalePrice'] = predictions
y_pred.to_csv('submission2.csv' , index = False )







