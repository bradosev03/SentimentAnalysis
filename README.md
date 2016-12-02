# Twitter Sentiment Analysis

### Brandon Radosevich
### NMSU Fall 2016 Semester
### EE 565/490: Pattern Recognition & Machine Learning
### Final Project


## training.py
```bash
$ python training.py -f linearsvc_model.pickle -t LinearSVC
$ python training.py -f bayesian_model.pickle -t Bayesian
$ python training.py -f svc_model.pickle -t SVC
$ python training.py -f decisiontree_model.pickle -t DecisionTree
```


## validation.py

```bash
$ python validation.py -f linearsvc_model.pickle
```

## twitter_handler.py

```bash
$ python twitter_handler.py -f linearsvc_model.pickle
```

#### File Structure:
```bash
.
├── bayesian_model.pickle
├── decisiontree_model.pickle
├── linearsvc_model.pickle
├── svc_model.pickle
├── training.py
├── twitter_classifer.py
├── twitter_handler.py
└── validation.py
```

 * [NLTK Library](http://www.nltk.org/) for Natural Langurage Processing