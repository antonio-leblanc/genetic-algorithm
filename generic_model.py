from funcoes_plot_new import *
from datetime import datetime
from dicionarios import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,explained_variance_score 
from sklearn.model_selection import GridSearchCV

# Modelos
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


#__________________________________GENERIC MODEL______________________________________#

class GenericModel:
    def __init__(self, x, y, Model, model_params={}, X_scaler = StandardScaler, Y_scaler = None, x_scaler_args ={},
                 y_scaler_args={}, tt_split_args={'test_size' : 0.2 , 'random_state' : 0, 'shuffle' : False}):
        
        # Scaling x and y
        self.x = x
        self.y = y
        if X_scaler:
            self.x_scaler = X_scaler(**x_scaler_args)
            self.x_scaler.fit(x)
            self.x_std = self.x_scaler.transform(x)
        else:
            self.x_std = x
            self.x_scaler = None
        
        if Y_scaler:
            self.y_scaler = Y_scaler(**y_scaler_args)
            self.y_scaler.fit(y)
            self.y_std = self.y_scaler.transform(y)
        else:
            self.y_std = y  
            self.y_scaler = None
            
        # Train Test Split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_std,self.y_std, **tt_split_args)
        # Model fit
        self.model = Model(**model_params)
        self.model.fit(self.x_train,self.y_train)
        # Model prediction
        self.y_prediction = self.model.predict(self.x_test)
    
    #______ Prediction for validation set ______
    def transform_predict(self,x_validation):
        return self.model.predict(self.x_scaler.transform(x_validation))
    
    #______GRID SEARCH______
    def grid_search(self,search_params,grid_params={}):
        print(f'Performing Grid Search\r') 
        start_timer = datetime.now()
        self.g_search = GridSearchCV(self.model, search_params,**grid_params)
        self.g_search.fit(self.x_train,self.y_train)
        print(f'Grid Search Completed | Time : {datetime.now()-start_timer}')
        print(f'•  Best Score : {self.g_search.best_score_:.4f}')
        print(f'• Best Params : {self.g_search.best_params_}')
        
    
    #______GETERS______
    def get_model(self):
        return self.model
    
    def get_x_scaler(self):
        return self.x_scaler
    
    def get_y_scaler(self):
        return self.y_scaler
    
    def get_test_score(self):
        return self.model.score(self.x_test,self.y_test)
    
    def get_train_score(self):
        return self.model.score(self.x_train,self.y_train)
    
    def get_test_prediction(self):
        return self.y_prediction
           
        
    #______ PRINT / PLOT METHODS______
    
    def print_eval_metrics(self,train=False):
        if not train:
            print ('→ Test metrics')
            print (f'• R² Score : {r2_score(self.y_test,self.y_prediction):.4f}')
            print (f'• MAE Score : {mean_absolute_error(self.y_test,self.y_prediction):.4f}')
            print (f'• RMSE Score : {np.sqrt(mean_squared_error(self.y_test,self.y_prediction)):.4f}')
            print (f'• % RMSE Score : {100*np.sqrt(mean_squared_error(self.y_test,self.y_prediction))/np.mean(self.y_test):.2f}')   
            print (f'• Explained variance score : {explained_variance_score(self.y_test,self.y_prediction):.4f}')
        else:
            train_prediction = self.model.predict(self.x_train)
            print ('→ Train metrics')
            print (f'• R² Score : {r2_score(self.y_train,train_prediction):.4f}')
            print (f'• MAE Score : {mean_absolute_error(self.y_train,train_prediction):.4f}')
            print (f'• RMSE Score : {np.sqrt(mean_squared_error(self.y_train,train_prediction)):.4f}')
            print (f'• % RMSE Score : {100*np.sqrt(mean_squared_error(self.y_train,train_prediction))/np.mean(self.y_train):.2f}')   
            print (f'• Explained variance score : {explained_variance_score(self.y_train,train_prediction):.4f}')
    
    def plot_line(self, start_idx=None, end_idx=None, title='', y_label='',size=(15,5)):
        plot_model_line(self.y_test,self.y_prediction,start_idx = start_idx, end_idx =end_idx,size=size,
                        title=title, y_label=y_label)
    
    def plot_scatter(self,title='',size=(6,6),s=2):
        plot_model_scatter(self.y_test,self.y_prediction,title=title,size=size,s=s)
        
    def plot_test_vars(self, start_idx=None, end_idx=None, size=(15,5)):
        start_idx = start_idx if start_idx else 0
        end_idx = end_idx if end_idx else len(self.x_test)
        pd.DataFrame(data=self.x_test,columns=self.x.columns).plot(figsize=size, lw=1);
    
    def plot_feature_importance(self):
        try:
            f_imp = pd.Series(self.model.feature_importances_,self.x.columns).sort_values()
            f_imp.plot(kind='barh',grid=True,title='Importância relativa das variáveis');
        except:
            print('Este modelo não tem "Feature Importance"')
        

