from sklearn.model_selection import train_test_split

def split(X, y, test_size=0.2):

    # get Test, set aside
    X_remainder, X_test, y_remainder, y_test = train_test_split(
        X, y, test_size=test_size
    )

    # get Train, Validate
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_remainder, y_remainder, test_size=test_size
    )

    return {
        # original
        'X': X,
        'y': y,

        # train test split
        'X_train': X_train,
        'y_train': y_train,
        'X_validate': X_validate,
        'y_validate': y_validate,
        'X_test': X_test,
        'y_test': y_test,

        # this is from the first split, can use cross validation
        'X_cv': X_remainder, 
        'y_cv': y_remainder
    }
