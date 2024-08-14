from sklearn.ensemble import VotingClassifier
   from sklearn.svm import SVC
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.neighbors import KNeighborsClassifier

   def create_ensemble():
       svm = SVC(probability=True)
       dt = DecisionTreeClassifier()
       knn = KNeighborsClassifier()
       
       ensemble = VotingClassifier(
           estimators=[('svm', svm), ('dt', dt), ('knn', knn)],
           voting='soft'
       )
       
       return ensemble

   # Create the ensemble model
   ensemble_model = create_ensemble()