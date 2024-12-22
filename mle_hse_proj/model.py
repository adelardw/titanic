from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class Model:
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
    
    def svm_fit_predict(self, svm_kwargs):
        
        model = SVC(**svm_kwargs)
        model.fit(self.X_train, self.y_train)
        self.predict = model.predict(self.X_test)
        
        return self.predict, self.y_test
    
    def decision_tree_fit_predict(self, tree_kwargs):
        
        model = DecisionTreeClassifier(**tree_kwargs)
        model.fit(self.X_train, self.y_train)
        self.predict = model.predict(self.X_test)
        
        return self.predict, self.y_test
    
    def lr_fit_predict(self, lr_kwargs):
        
        model = LogisticRegression(**lr_kwargs)
        model.fit(self.X_train, self.y_train)
        self.predict = model.predict(self.X_test)
        
        return self.predict, self.y_test
    