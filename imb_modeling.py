import pandas as pd
# scikit-learn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split,RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#imblearn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

def imb_pipe_fit(model, params, X, y, score='roc_auc', scaler=StandardScaler()):
    """
    utilize imblearn to create pipeline(smote->scale->model) and do gridsearch on model
    @params: model to fit,
             param_grid dictionary, 
             predictors(X), 
             response(y), 
             score_metric (sklearn), https://scikit-learn.org/stable/modules/model_evaluation.html
             scaler
    @return: dict of gridsearch best param model, cv_score, test_score
    """

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        stratify=y,
                                                        random_state=421)

    pipeline = Pipeline(steps = [['smote', SMOTE(random_state=421)],
                                 ['scaler', scaler],
                                 ['model', model]]
                        )

    folds = RepeatedStratifiedKFold(n_splits=10, 
                                               n_repeat=5,
                                               shuffle=True,
                                               random_state=421
                                               )   
    
    gs = GridSearchCV(estimator=pipeline,
                               param_grid=params,
                               scoring=str(score),
                               cv=folds,
                               refit=True,
                               n_jobs=-1)
    
    gs.fit(X_train, y_train)
    cv_score = gs.best_score_
    test_score = gs.score(X_test, y_test)
    return {'model':gs, 'cv_score':cv_score, 'test_score':test_score}