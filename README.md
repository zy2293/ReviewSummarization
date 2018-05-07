# ReviewSummarization

Goal: Summarize features of products(restaurants) and customers' sentiments over these features from customers reviews

Output format: feature, opinion, score. Eg: (bagel, excellent, 1.25)

Preprocess:
lang.py: 
- select language
- filter stop_words
- remove inline punctuations: eg: soy/\sauce
- lowercase
- singularize nouns
- Part-of-Speech(PoS) tag reviews


Two Methods:

1.
generateFeature.py: use Associate Rule Mining to generate frequent features list for each product
generateScore.py: summarize opinions towards features and compute setniment scores

2. 
DependencyParse.py: apply "amod" (adjective-modified) mode available in "Spacy" package to identify feature-opinion pairs
