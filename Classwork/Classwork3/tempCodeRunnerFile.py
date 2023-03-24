rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print(rf.get_params())
print(rf.score(X_test, y_test))