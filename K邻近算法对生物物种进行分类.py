from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report


iris = load_iris()
#查看数据说明
print(iris.DESCR)
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size = 0.25, random_state=33)
ss = StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.fit_transform(X_test)
knc=KNeighborsClassifier()
knc.fit(X_train, Y_train)
Y_predict=knc.predict(X_test)

print("K分类法准确率为：", knc.score(X_test, Y_test))
print(classification_report(Y_test, Y_predict, target_names=iris.target_names))
