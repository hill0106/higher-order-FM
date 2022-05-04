from tensorfm.sklearn import FactorizationMachineRegressor
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction import DictVectorizer
import numpy as np



# Read in data
def load_data(filename, path="examples/dataset/ml-100k/"):
    data = []
    y = []
    users=set()
    items=set()
    genre=set()
    with open(path+filename) as f:
        for line in f:
            (user,movieid,rating,ts)=line.split('\t')
            data.append({ "user_id": str(user), "movie_id": str(movieid)})#, "timestamp": str(ts)})
            y.append(float(rating))
            users.add(user)
            items.add(movieid)
            #timestamp.add(ts)
    return (data, np.array(y), users, items)#, timestamp)


train_data, y_train, train_users, train_items= load_data("ua.base")
test_data, y_test, test_users, test_items = load_data("ua.test")
v = DictVectorizer()
X_train = v.fit_transform(train_data)
X_test = v.transform(test_data)


y_train.shape += (1,)
# print(y_train)
# print(X_test.todense().shape)
fm = FactorizationMachineRegressor(max_iter=100, n_factors=3, eta=0.01, C=10000, random_state=12345, penalty='l2')
fm.fit(X_train.todense(), y_train.ravel())
y_pred = fm.predict(X_test.todense())
print(y_pred)
print("MSE test ", mean_squared_error(y_test, y_pred))
