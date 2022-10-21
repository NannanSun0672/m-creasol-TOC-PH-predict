"""
Create on May 21,2020

@author:nannan.sun

Function:


1.交互项

2.使用岭回归，Lasso 回归做污水 TOC去除率（％），IP转化率（％），以及反应后pH值预测

"""
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures,MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib as mpl
import warnings
import pandas as pd
class Polluted_predict(object):
    def __init__(self):
        self.data_excel_path = "/Users/sunnannan/Desktop/数据.xlsx"
        self.x, self.y_TOC,self.y_IP,self.y_pH = self.data_preprocess()
        self.models()

    def data_preprocess(self):
        """
        1.归一化数据
        :return:
        """
        data_info = pd.read_excel(self.data_excel_path)
        features = ["T","P","t","m","H"]
        #data_info_train = data_info.ix[:,:]
        #data_info_val = data_info.ix[38:,:]
        #print(data_info_val)
        X = data_info[features]
        y_TOC = data_info["TOC"]
        y_IP = data_info["IP"]
        y_pH = data_info["pH"]
        #X_val = data_info_val[features]
        #y_TOC_val = data_info_val["TOC"]
        #y_IP_Val = data_info_val["IP"]
        #y_pH_Val = data_info_val["pH"]
        #x_train,x_val,y_train,y_val = train_test_split(X,y_TOC, test_size=0.1, random_state=101)
        min_max_scaler = MinMaxScaler()
        #print(X.shape)
        x_scaler = min_max_scaler.fit_transform(X)
        #x_Val_scaler = min_max_scaler.fit_transform(X_val)
        #x_val_scaler = min_max_scaler.fit_transform(x_val)
        #print(x_scaler)

        return x_scaler, y_TOC, y_IP, y_pH

    def xss(self,y, y_hat):
        y = y.ravel()
        y_hat = y_hat.ravel()
        # Version 1
        tss = ((y - np.average(y)) ** 2).sum()
        rss = ((y_hat - y) ** 2).sum()
        ess = ((y_hat - np.average(y)) ** 2).sum()
        r2 = 1 - rss / tss
        # print 'RSS:', rss, '\t ESS:', ess
        # print 'TSS:', tss, 'RSS + ESS = ', rss + ess
        #tss_list.append(tss)
        #rss_list.append(rss)
        #ess_list.append(ess)
        #ess_rss_list.append(rss + ess)
        # Version 2
        # tss = np.var(y)
        # rss = np.average((y_hat - y) ** 2)
        # r2 = 1 - rss / tss
        corr_coef = np.corrcoef(y, y_hat)[0, 1]
        return r2, corr_coef


    def models(self):
        """
        1.建立线性模型
        :return:
        """
        models = [Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', LinearRegression(fit_intercept=False))]),
            Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', RidgeCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),

            Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', LassoCV(alphas=np.logspace(-3, 2, 50), fit_intercept=False))]),
            Pipeline([
                ('poly', PolynomialFeatures()),
                ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 50), l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                        fit_intercept=False))])

            ]
        d_pool = np.arange(1, 8, 1)  # 阶
        titles = u'线性回归',u'Ridge回归', u'LASSO', u'ElasticNet'
        #param_grid = {"alpha": alpha_range}
        for t in range(4):
            model = models[t]
            for i, d in enumerate(d_pool):
                model.set_params(poly__degree=d)
                #scores = cross_val_score(model, self.x, self.y_pH.ravel(), cv=10)
                model.fit(self.x, self.y_pH.ravel())
                lin = model.get_params('linear')['linear']

                #print(lin)
                output = u'%s：%d阶，系数为：' % (titles[t], d)
                if hasattr(lin, 'alpha_'):
                    idx = output.find(u'系数')
                    output = output[:idx] + (u'alpha=%.6f，' % lin.alpha_) + output[idx:]
                if hasattr(lin, 'l1_ratio_'):  # 根据交叉验证结果，从输入l1_ratio(list)中选择的最优l1_ratio_(float)
                    idx = output.find(u'系数')
                    output = output[:idx] + (u'l1_ratio=%.6f，' % lin.l1_ratio_) + output[idx:]
                print(output, len(lin.coef_.ravel()))

                #s = model.score(self.x_val, self.y_TOC_val)
                r2, corr_coef = self.xss(self.y_pH, model.predict(self.x))
                #r2_val, corr_coef_val = self.xss(self.y_pH_val, model.predict(self.x_val))
                print(output,"训练集上R2",r2,"十折交叉验证平均R2")
                # print 'R2和相关系数：', r2, corr_coef
                # print 'R2：', s, '\n'
if __name__ == "__main__":
    Polluted_predict = Polluted_predict()


