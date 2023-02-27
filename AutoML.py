from catboost import CatBoostClassifier
from pycaret.classification import *
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import collections
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class Score:
    def __init__(self):
        self.rf = None
        self.X_test = None
        self.y_test = None
        self.dataset = None
        self.catboost = None
        self.light = None
        self.xgboost = None
        self.ensemble = None
        self.loaded_model = None
        self.test_set = None
        self.target = 'is_bad30'

    def dataset_setup(self):
        print('Setup...')
        train, test = self.prep_dataset()

        setup(target=self.target, silent=True, html=False, data=train, test_data=test, data_split_shuffle=True,
              numeric_features=['amount_smn', 'duration', 'credit_history_count', 'gender', 'age', 'dependants',
                                'mon_remit', 'int_rate', 'mon_payment', 'int_amount'],
              categorical_features=['marital_status', 'district', 'sector'],
              ignore_features=['currency'])

    def prep_dataset(self):
        # train = pd.read_excel('datasets/Spitamen/SPay/Spay predict/main_merged_train.xlsx', index_col=False, engine='openpyxl')
        # test = pd.read_excel('datasets/Spitamen/SPay/Spay predict/main_merged_test.xlsx', index_col=False, engine='openpyxl')
        dataset = pd.read_csv('datasets/dataset1-6-trade.csv', index_col=False)

        '''
        variables_aab = ['Region', 'Month', 'Week_Eng', 'MonPay', 'IntIncome', 'Age', 'Gender', 'Amount',
                         'DurationReal', 'IntRate', 'MaritalStatus', 'Income', 'Dependants', 'ResPeriod',
                         'OccupationBranch', 'Occupation', 'ExpCat', 'CarOwner', 'HouseOwner', 'FOOD', 'NONFOOD',
                         'SERVICES', 'EXRATE', 'AVRATE', 'PastCreditType', 'CreditAccCount', 'GDP', self.target,
                         'IsPandemic']
        variables_spay = ['gender', 'int_rate', 'amount_smn', 'duration',  'mon_payment', 'int_amount',
                          'sector', 'credit_goal', 'district',
                          'CBT Фаври_sum', 'IBT Visa_sum', 'IBT Корти милли_sum', 'Алиф Сармоя Карта_sum',
                          'Коммерсбонки Точикистон_sum', 'Корти Милли_sum', 'МДО FINCA_sum',
                          'МЗО ХУМО и Партнёры_sum', 'МИР/VISA/MasterCard РФ_sum', 'Спитамен Банк Visa_sum',
                          'Тезинфоз_sum', 'Komyob_sum', 'Pari Match_sum', 'Tennisi.tj_sum', 'Формул_sum', 'Формула 55_sum',
                          'FMFB Credit_sum', 'IBT Credit_sum', 'Алиф Сармоя Кредит_sum', 'Банк Арванд Кредит_sum',
                          'Спитамен Банк Кредит_sum', 'Электричество Бохтар_sum', 'Электричество Бустон_sum', 'Электричество Гулистон_sum',
                          'Электричество Душанбе_sum', 'Электричество Зафарабад_sum', 'Электричество Истаравшан_sum',
                          'Электричество Турсунзаде_sum', 'Электричество Хисор_sum', 'Электричество Худжанд_sum', 'DC Wallet_sum', 'ExpressPay_sum', 'IBT WALLET_sum', 'Megafon Life_sum',
                          'QIWI Кошелек (Таджикистан)_sum', 'QIWI Кошелек_sum', 'Алиф Сармоя Кошелек_sum',
                          'Амонатбанк Кошелек_sum', 'Банк Арванд Кошелек_sum', 'ВКонтакте_sum',
                          'ФАБЕРЛИК-пополнение счета_sum', 'Хумо Онлайн_sum', 'ЮMoney (Яндекс.Деньги)_sum', 'Warface_sum', 'Arzon.tj_sum', 'Echipta_sum', 'RG.TJ_sum', 'Somon.tj_sum',
                          'QR Платеж_sum', 'Tajrupt_sum', 'TaxiOlucha основной счёт_sum', 'IRS_sum',
                          'Beeline-Tajikistan Таджикистан_sum', 'NGN Вавилон_sum', 'Omobile_sum',
                          'Tcell (Таджикистан)_sum', 'Tcell_sum', 'Tele2 Казахстан_sum', 'TELE2_sum', 'TezNet_sum',
                          'TojNET_sum', 'UzMobile (Узбекистан)_sum', 'Yota_sum', 'ZetMobile(Билайн)_sum',
                          'ZET-MOBILE_sum', 'АНТ-ТВ_sum', 'Билайн Кыргызстан_sum', 'Билайн Россия_sum',
                          'Билайн РФ_sum', 'Билайн-Узбекистан_sum', 'Вавилон Мобайл_sum',
                          'Все операторы Кыргыстана_sum', 'Интернет Вавилон-Т_sum', 'Мавчи Сомон_sum',
                          'Мегафон Россия_sum', 'МегаФон РФ_sum', 'Мегафон Таджикистан_sum', 'МТС Россия_sum',
                          'МТС РФ_sum', 'Оила ТВ_sum', 'Теле 2_sum', 'Точик Телеком_sum', 'ТТL Интернет_sum',
                          'Водоканал (Юг)_sum', 'Водоканал г.Душанбе_sum', 'Водоканал Душанбе_sum',
                          self.target]
        variables_ibt = ['age', 'sector', 'gender', 'amount_smn', 'duration', 'int_rate',
                         'mon_payment', 'int_amount', 'marital_status', 'district', 'credit_history_count', 'mon_remit',
                         self.target]
        variables_mycar = ['amount_issued', 'car_price', 'initial_fee', 'ltv', 'duration', 'has_coborrower', 'gender',
                           'address', 'age', 'marital_status', 'income', 'status',
                           self.target]
        variables_imon = ['amount_smn', 'duration', 'credit_history_count', 'gender', 'age', 'dependants',
                          'marital_status', 'district', 'sector',
                          'usd_rub_daily', 'neft_price', 'mon_remit',
                          'int_rate', 'mon_payment', 'int_amount',
                          'date_issue',
                          self.target]
        variables_commerce = ['age', 'gender', 'amount_smn', 'duration', 'int_rate', 'credit_history_count',
                              'mon_remit', 'mon_payment', 'int_amount',
                              'sector', 'marital_status', 'district',
                              self.target]
        variables_spay = ['duration', 'district', 'currency', 'sector', 'int_rate', 'amount_smn',
                          'banking_amount', 'banking_count', 'betting_amount', 'betting_count', 'utilities_amount',
                          'utilities_count', 'wallet_amount', 'wallet_count', 'online_market_amount',
                          'online_market_count', 'service_amount', 'service_count', 'telecom_amount', 'telecom_count',
                          self.target]
        variables_score2 = ['amount_smn', 'duration', 'credit_history_count', 'gender', 'age',
                            'marital_status', 'district', 'sector', 'currency', 'dependants',
                            'mon_remit', 'avg_remit',
                            'usd_rub_daily', 'neft_price',
                            'int_rate', 'mon_payment', 'int_amount',
                            'version',
                            self.target]
        variables_swiss = ['amount', 'duration', 'collateral', 'LTV', 'branch', 'co_borrower', 'gender', 'age',
                           'food', 'nonfood', 'services', 'avrate', 'gdp', 'marital_status',
                           self.target]
        variables_dc = ['amount_smn', 'banking_amount', 'betting_amount', 'district', 'duration', 'gender',
                        'int_amount', 'int_rate', 'mon_payment', 'sector', 'telecom_amount', 'utilities_amount',
                        'wallet_amount', self.target]
        '''

        variables_score2 = ['amount_smn', 'duration', 'credit_history_count', 'gender', 'age', 'dependants',
                            'marital_status', 'district', 'sector',
                            'mon_remit', 'int_rate', 'mon_payment', 'int_amount',
                            'currency',
                            self.target]

        dataset = dataset[variables_score2]
        dataset = dataset[dataset['currency'] <= 'TJS']
        dataset = dataset[dataset['amount_smn'] <= 30000]
        dataset = dataset[(dataset['age'] >= 18) & (dataset['age'] <= 80)]
        dataset.dropna(inplace=True)

        train, test = train_test_split(dataset, test_size=0.15)
        # train.to_excel('datasets/mycar/mycar_closedonly_trainset.xlsx', index=False)
        # test.to_excel('datasets/mycar/mycar_closedonly_testset.xlsx', index=False)

        print('All samples:',           len(train) + len(test))
        print('Train Good samples:',    len(train[train[self.target] == 0]))
        print('Train Bad samples:',     len(train[train[self.target] == 1]))
        print('Test Good samples:',     len(test[test[self.target] == 0]))
        print('Test Bad samples:',      len(test[test[self.target] == 1]))

        return train, test

    def train_rf(self):
        print('RF training... ')

        self.rf = create_model(RandomForestClassifier(criterion='entropy', max_depth=12, max_features=8,
                                                      max_leaf_nodes=None, max_samples=None, min_impurity_split=None,
                                                      min_samples_leaf=65, min_samples_split=10,
                                                      n_estimators=150, verbose=0),
                               fold=5)

        print('RF training Complete!')
        return self.rf

    def train_lightgbm(self):
        print('LightGBM training... ')
        self.light = create_model('lightgbm', n_estimators=900, max_depth=8, num_leaves=70, min_data_in_leaf=500)
        print('LightGBM training Complete!')
        return self.light

    def train_catboost(self):
        print('Catboost training... ')

        self.catboost = create_model(CatBoostClassifier(learning_rate=0.02, depth=7, l2_leaf_reg=30,
                                                        random_strength=0.7, n_estimators=1000, eval_metric='AUC'),
                                     fold=5, verbose=False)
        print('Catboost training Complete!')
        return self.catboost

    def train_xgboost(self):
        print('Xgboost training... ')
        self.xgboost = create_model(XGBClassifier(eval_metric='auc'))
        '''
        self.xgboost = create_model(XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                                    colsample_bynode=1, colsample_bytree=0.5, gamma=0, gpu_id=-1,
                                    importance_type='gain', interaction_constraints='',
                                    learning_rate=0.1, max_delta_step=0, max_depth=11,
                                    min_child_weight=1, monotone_constraints='()',
                                    n_estimators=140, n_jobs=6, num_parallel_tree=1,
                                    objective='binary:logistic', random_state=0, reg_alpha=0.001,
                                    reg_lambda=4, scale_pos_weight=1.9, subsample=0.7,
                                    tree_method='exact', use_label_encoder=False,
                                    validate_parameters=1, verbosity=None))
        '''
        print('Xgboost training Complete!')
        return self.xgboost

    def train_ensemble(self, model1, model2):
        print('Blending models...')
        self.ensemble = blend_models([model1, model2], fold=5, method='soft')
        print('Blending Complete!')
        return self.ensemble

    @staticmethod
    def plot_feat_importance(model):
        plot_model(model, plot='feature_all')
        '''
        cats = ['maritalStatus', 'district', 'sector', 'currency', 'season', 'month']
        imp = pd.DataFrame({'Feature': model.feature_names_,
                            'Value': abs(model.feature_importances_)})

        for column in cats:
            cat = imp[imp['Feature'].str.contains(column, na=False)]
            summ = cat['Value'].sum()
            imp = imp[~imp['Feature'].str.contains(column, na=False)]
            imp = imp.append({'Feature': column,
                              'Value': summ}, ignore_index=True)

        imp.sort_values(by='Value', ascending=True, inplace=True)
        imp.plot.barh(y='Value', x='Feature')
        print('SUMMM:', imp['Value'].sum())
        plt.tight_layout()
        plt.show()
        # imp.to_excel('imp_full.xlsx')'
        '''

    def predict_ml(self, model, threshold=0.2):
        print('Predicting...')
        pd.set_option('display.max_columns', None)
        predictions = predict_model(model, raw_score=True, encoded_labels=True)
        predictions['Label'] = (predictions['Score_1'] > threshold).astype(int)
        report = classification_report(predictions[self.target], predictions['Label'], output_dict=True)
        tn, fp, fn, tp = confusion_matrix(predictions[self.target], predictions['Label']).ravel()
        auc = roc_auc_score(predictions[self.target], predictions['Score_1'])
        report['threshold'] = threshold
        report['TN'] = tn
        report['FP'] = fp
        report['FN'] = fn
        report['TP'] = tp
        report['AUC'] = auc
        print('Prediction Complete!')

        return report

    @staticmethod
    def predict_sample(model, sample, threshold=0.15):
        predictions = predict_model(model, raw_score=True, data=sample, encoded_labels=True)
        predictions['Label'] = (predictions['Score_1'] > threshold).astype(int)
        print(predictions.info())
        return predictions[['Score_0', 'Label']]

    def test_ml(self, model, path, threshold=0.15):
        print('Predicting...')
        dataset = pd.read_excel(path, index_col=False)
        pd.set_option('display.max_columns', None)
        predictions = predict_model(model, data=dataset, raw_score=True, encoded_labels=True)
        predictions['Label'] = (predictions['Score_1'] > threshold).astype(int)
        report = classification_report(predictions[self.target], predictions['Label'], output_dict=True)
        tn, fp, fn, tp = confusion_matrix(predictions[self.target], predictions['Label']).ravel()
        auc = roc_auc_score(predictions[self.target], predictions['Score_1'])
        report['threshold'] = threshold
        report['TN'] = tn
        report['FP'] = fp
        report['FN'] = fn
        report['TP'] = tp
        report['AUC'] = auc
        print('Prediction Complete!')

        return report

    @staticmethod
    def load_ml(load_path):
        return load_model(load_path)

    @staticmethod
    def save_ml(model, model_name):
        # model_name = input('Model save filename: ')
        if model_name:
            print('Saving model ' + model_name + '...')
            save_model(model, 'models/' + model_name)

    def threstest(self, model, filename):
        results = pd.DataFrame(columns=['Accuracy', 'AUC', 'TN', 'FP', 'FN', 'TP', 'threshold'])
        predictions = predict_model(model, raw_score=True, encoded_labels=True)

        for thres in [x / 100 for x in range(10, 26, 1)]:
            predictions['Label'] = (predictions['Score_1'] > thres).astype(int)
            tn, fp, fn, tp = confusion_matrix(predictions[self.target], predictions['Label']).ravel()
            auc = roc_auc_score(predictions[self.target], predictions['Score_1'])
            accuracy = accuracy_score(predictions[self.target], predictions['Label'])

            row = [accuracy, auc, tn, fp, fn, tp, thres]
            results.loc[len(results)] = row

        print(results)

        results.to_excel('outputs/'+filename+'.xlsx')

    @staticmethod
    def final_model(model):
        fin_model = finalize_model(model)
        return fin_model
