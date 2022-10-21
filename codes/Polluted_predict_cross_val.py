"""
Create on May 21,2020

@author:nannan.sun@wowjoy.cn

Function:


1.预处理数据

2.使用岭回归，Lasso 回归做污水 TOC去除率（％），IP转化率（％），以及反应后pH值预测

"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge,Lasso
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import sklearn.metrics as sm
from sklearn.neural_network import MLPRegressor

class Polluted_predict(object):
    def __init__(self):
        self.data_path = "/Users/sunnannan/Desktop/数据.txt"
        self.data_excel_path = "/Users/sunnannan/Desktop/数据.xlsx"
        self.x, self.y_TOC,self.y_IP,self.y_pH = self.data_preprocess()
        self.Ridge_model()
        #self.Lasso_model()
        #self.ANN_model()

    def data_preprocess(self):
        """
        1.归一化数据
        :return:
        """
        data_info = pd.read_excel(self.data_excel_path)
        features = ["T","P","t","m","H"]
        X = data_info[features]
        y_TOC = data_info["TOC"]
        y_IP = data_info["IP"]
        y_pH = data_info["pH"]
        #x_train,x_val,y_train,y_val = train_test_split(X,y_TOC, test_size=0.1, random_state=101)
        min_max_scaler = preprocessing.MinMaxScaler()
        #print(X.shape)
        x_scaler = min_max_scaler.fit_transform(X)
        #x_val_scaler = min_max_scaler.fit_transform(x_val)
        #print(x_scaler)

        return x_scaler, y_TOC, y_IP, y_pH

    def Ridge_model(self):
        """
        1.建立线性回归模型
        :return:
        """
        kfold = KFold(n_splits=10)
        alpha_range = alphas=np.logspace(-3, 2, 50)  # 生成alpha测试集
        param_grid = {"linear__alpha": alpha_range}
        Ridge_model = Ridge(fit_intercept=False)
        pipe = Pipeline([('poly', preprocessing.PolynomialFeatures()),("linear",Ridge_model)])
        d_pool = np.arange(1, 8, 1)
        for i, d in enumerate(d_pool):
            pipe.set_params(poly__degree=d)
            grid_search = GridSearchCV(pipe,param_grid,cv=kfold)
        #print(self.x_pH.shape)
            grid_search.fit(self.x,self.y_pH)

            print(grid_search.cv_results_["mean_test_score"])

        #pred_y = grid_search.best_estimator_.predict(self.x_val)

        #import IPython
        #IPython.embed()
        #pred_y = grid_search.best_estimator_.predict(self.x_pH)
        #print(pred_y)
        #print('平均绝对值误差：', sm.mean_absolute_error(self.y_pH, pred_y))
        #print('平均平方误差：', sm.mean_squared_error(self.y_pH, pred_y))
        #print('中位绝对值误差：', sm.median_absolute_error(self.y_pH, pred_y))
        #print('R2得分：', sm.r2_score(self.y_val_TOC, pred_y))


    def Lasso_model(self):
        """
        1.
        :return:
        """
        kfold = KFold(n_splits=10)
        alpha_range = alphas = np.logspace(-3, 2, 50)  # 生成alpha测试集
        param_grid = {"linear__alpha": alpha_range }
        Lasso_model = Lasso(normalize=False)
        # Ridge_model.fit(self.x,self.y_TOC)
        pipe = Pipeline([('poly', preprocessing.PolynomialFeatures()), ("linear", Lasso_model)])
        d_pool = np.arange(1, 8, 1)
        for i, d in enumerate(d_pool):
            pipe.set_params(poly__degree=d)
            grid_search = GridSearchCV(pipe, param_grid, cv=kfold)
            # print(self.x_pH.shape)
            grid_search.fit(self.x, self.y_pH)

            print(grid_search.cv_results_["mean_test_score"])

        #print(grid_search.best_score_)
        #pred_y = grid_search.best_estimator_.predict(self.x)
        #print(pred_y)
        #print('平均绝对值误差：', sm.mean_absolute_error(self.y_pH, pred_y))
        #print('平均平方误差：', sm.mean_squared_error(self.y_pH, pred_y))
        #print('中位绝对值误差：', sm.median_absolute_error(self.y_pH, pred_y))
        #print('R2得分：', sm.r2_score(self.y_pH, pred_y))

    def ANN_model(self):
        """
        1.
        :return:
        """
        kfold = KFold(n_splits=10)
        alpha_range = np.linspace(0.00001, 0.0001, 30)  # 生成alpha测试集
        #alpha_range = alphas = np.logspace(-3, 2, 50)
        param_grid = {"alpha":alpha_range}

        clf = MLPRegressor(solver='lbfgs',hidden_layer_sizes=(5,2), random_state=1)
        #pipe = Pipeline([('poly', preprocessing.PolynomialFeatures()), ("linear", clf)])
        #d_pool = np.arange(1, 2, 1)
        #for i, d in enumerate(d_pool):
        #    print(i,d)
        #    pipe.set_params(poly__degree=d)
        #    grid_search = GridSearchCV(pipe, param_grid, cv=kfold)
            # print(self.x_pH.shape)
        #    grid_search.fit(self.x, self.y_TOC)

        #    print(grid_search.cv_results_["mean_test_score"])
        grid_search = GridSearchCV(clf, param_grid)
        print(self.x.shape)
        grid_search.fit(self.x, self.y_pH)
        print(grid_search.cv_results_)
        #clf.fit(self.X, self.y)







    def data_Trans(self):
        """
        1.转换 DataFrame
        :return:
        """
        ######特征
        T_list = []
        P_list = []
        t_list = []
        m_list = []
        H_list = []
        #######预测目标
        TOC_list = []
        IP_list = []
        pH_list = []
        with open(self.data_path, encoding = 'gb2312') as fr:
            content = fr.readlines()
        print(content)
        for idx,line in enumerate(content):
            if "T" in line and "O" not in line:
                line_splited = line.split()
                #print(len(line_splited))
                T_list.extend(line_splited[1:])
            elif "P" in line and "I" not in line:
                line_splited = line.split()
                P_list.extend(line_splited[1:])
            elif "t"in line:
                line_splited = line.split()
                t_list.extend(line_splited[1:])
            elif "m" in line:
                line_splited = line.split()
                #print(len(line_splited))
                m_list.extend(line_splited[1:])
            elif "H" in line and "p" not in line:
                line_splited = line.split()
                #print(len(line_splited))
                H_list.extend(line_splited[1:])
            elif "TOC" in line:
                line_splited = line.split()
                #print(line_splited)
                TOC_list.extend(line_splited[1:])
            elif "IP" in line:
                line_splited = line.split()
                #print(line_splited)
                IP_list.extend(line_splited[1:])
            elif "pH" in line:
                line_splited = line.split()
                #print(len(line_splited))
                pH_list.extend(line_splited[1:])

        data_dict = {"T":T_list,"P":P_list,"t":t_list,"m":m_list,"H":H_list,"TOC":TOC_list,"IP":IP_list,"pH":pH_list}
        #print(data_dict)
        data_frame = pd.DataFrame(data_dict)
        #print(data_frame)
        data_frame.to_excel("/Users/sunnannan/Desktop/污水/数据.xlsx")



if __name__ == "__main__":
    Polluted_predict = Polluted_predict()
