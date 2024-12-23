from data import Transforms
from model import Model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

def parse_kwargs(kwargs_list):
    kwargs_dict = {}
    for arg in kwargs_list:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                
                value = eval(value)
            except (NameError, SyntaxError):
                pass 
            kwargs_dict[key] = value
    return kwargs_dict    


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_csv_path', required=True)
    parser.add_argument('--test_csv_path', required=False, default=None)
    parser.add_argument('--feature_columns', nargs="+", default=["Pclass", "Sex", "Age"], 
                        help="Columns to use from the dataset (e.g., Pclass Sex Age)")
    parser.add_argument('--target_column', required=True)
    parser.add_argument('--model', choices=["svm", "decision_tree", "lr"], help="Choose a model to use: svm, decision_tree, or lr")
    parser.add_argument('--model_kwargs', nargs="*", help="Named arguments for the model in the format key=value", default=[])
    
    return parser.parse_args()


def main():
    args = build_parser()
    feature_columns = args.feature_columns
    target = [args.target_column]

    columns = feature_columns + target
    
    
    tf = Transforms()
    if args.test_csv_path:
        
        train = pd.read_csv(args.train_csv_path)
        test = pd.read_csv(args.test_csv_path)
        passenger_id = test['PassengerId']
        
        prepared_train = tf.fill_na(
                        tf.all_cat_column_label_encode(
                        tf.select_columns(df=train, features=columns) ),
                        column='Age', value=train['Age'].median())
        
        if args.target_column not in list(test.columns):
            columns = feature_columns
            y_test = None

        prepared_test = tf.fill_na(
                        tf.all_cat_column_label_encode(
                        tf.select_columns(df=test, features=columns)))
        
        
        X_train = prepared_train[feature_columns]
        X_test = prepared_test[feature_columns]
        y_train = prepared_train[target]
        
        
        
    else:
        dset = pd.read_csv(args.train_csv_path)
        prepared_dset = tf.fill_na(
                        tf.all_cat_column_label_encode(
                        tf.select_columns(df=dset, features=columns) ),
                        column='Age', value=dset['Age'].median())

        X = prepared_dset[feature_columns]
        y = prepared_dset[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model = Model(X_train, X_test, y_train, y_test)
    
    model_kwargs = parse_kwargs(args.model_kwargs)
    if args.model == 'svm':
        predict, y_test = model.svm_fit_predict(model_kwargs) 

    if args.model == 'decision_tree':
        predict, y_test = model.decision_tree_fit_predict(model_kwargs) 

    if args.model == 'lr':
        predict, y_test = model.lr_fit_predict(model_kwargs) 

    if y_test is not None:
        print(classification_report(y_test, predict))
    else:
        print('=' * 30, 'Saving answer ...', '='*30)
        pd.DataFrame(zip(passenger_id, predict.tolist())).to_csv(f'answer_{args.model}.csv', index=False)
        print('=' * 30, 'Great! Check the Answer', '='*30)
        
    

if __name__=='__main__':
    main()
